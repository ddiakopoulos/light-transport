#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "util.hpp"
#include "bvh.hpp"
#include "sampling.hpp"
#include "bsdf.hpp"
#include "lights.hpp"
#include "objects.hpp"
#include "math-util.hpp"
#include "gl-api.hpp"
#include "gl-imgui.hpp"

using namespace gui;

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

// Reference
// http://graphics.pixar.com/library/HQRenderingCourse/paper.pdf
// http://fileadmin.cs.lth.se/cs/Education/EDAN30/lectures/S2-bvh.pdf
// http://www.cs.utah.edu/~edwards/research/mcRendering.pdf
// http://computergraphics.stackexchange.com/questions/2130/anti-aliasing-filtering-in-ray-tracing
// http://web.cse.ohio-state.edu/~parent/classes/782/Lectures/05_Reflectance_Handout.pdf
// http://www.cs.cornell.edu/courses/Cs4620/2013fa/lectures/22mcrt.pdf
// http://cg.informatik.uni-freiburg.de/course_notes/graphics2_08_renderingEquation.pdf
// http://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
// http://mathinfo.univ-reims.fr/IMG/pdf/Using_the_modified_Phong_reflectance_model_for_Physically_based_rendering_-_Lafortune.pdf
// http://www.rorydriscoll.com/2009/01/07/better-sampling/
// http://www.cs.cornell.edu/courses/cs465/2004fa/lectures/22advray/22advray.pdf
// http://vision.ime.usp.br/~acmt/hakyll/assets/files/wynn.pdf
// https://graphics.stanford.edu/courses/cs348b-03/papers/veach-chapter9.pdf

// ToDo
// ----------------------------------------------------------------------------
// [X] Decouple window size / framebuffer size for gl render target
// [X] Whitted Raytraced scene - spheres with phong shading
// [X] Occlusion support
// [X] ImGui Controls
// [X] Add tri-meshes (Shaderball, lucy statue from *.obj)
// [X] Path tracing (Monte Carlo) + Jittered Sampling
// [X] Ray Antialiasing (tent filtering)
// [X] Timers for various functions (accel vs non-accel)
// [X] Proper radiance based materials (bdrf)
// [X] BVH Accelerator
// [X] Fix Intersection / Occlusion Bug
// [X] Lambertian Material / Diffuse + Specular + Transmission Terms
// [X] Mirror + Glass Materials
// [\] Area Lights
// [ ] Cook-Torrance Microfacet BSDF implementation
// [ ] Sampling Scheme(s): tiled, lines, random, etc
// [ ] Cornell scene loader, texture mapping & normals
// [ ] Alternate camera models: pinhole, fisheye, spherical
// [ ] Add other primitives (box, plane, disc)
// [ ] Skybox Sampling
// [ ] Realtime GL preview
// [ ] Portals (hehe)
// [ ] Bidirectional path tracing
// [ ] Other render targets: depth buffer, normal buffer
// [ ] Embree acceleration

template<class T> 
inline void flip_vertical_inplace(std::vector<T> & image, int2 size)
{
    for (int y = 0; y < size.y / 2; y++)
    {
        for (int x = 0; x < size.x; x++)
        {
            std::swap(image[y*size.x + x], image[(size.y - 1 - y)*size.x + x]);
        }
    }
}

inline bool take_screenshot(int2 size)
{
    HumanTime t;
    std::vector<uint8_t> screenShot(size.x * size.y * 3);
    glReadPixels(0, 0, size.x, size.y, GL_RGB, GL_UNSIGNED_BYTE, screenShot.data());
    for (int y = 0; y < size.y; ++y) std::memcpy(screenShot.data() + y * size.x * 3, screenShot.data() + (size.y - y - 1)*size.x * 3, size.x * 3);
    stbi_write_png(std::string("render_" + t.make_timestamp() + ".png").c_str(), size.x, size.y, 3, screenShot.data(), 3 * size.x);
    return false;
}

///////////////
//   Scene   //
///////////////

struct Scene
{
    float3 environment;
    float3 ambient;

    std::vector<std::shared_ptr<Traceable>> objects;
    std::vector<std::shared_ptr<Light>> lights;

    std::unique_ptr<BVH> bvhAccelerator;

    void accelerate()
    {
        bvhAccelerator.reset(new BVH(objects));
        bvhAccelerator->build();
        bvhAccelerator->debug_traverse(bvhAccelerator->get_root());
    }

    const int maxRecursion = 5;

    RayIntersection scene_intersects(const Ray & ray)
    {
        RayIntersection isct;
        if (bvhAccelerator)
        {
            isct = bvhAccelerator->intersect(ray);
        }
        else
        {
            for (auto & obj : objects)
            {
                const RayIntersection hit = obj->intersects(ray);
                if (hit.d < isct.d) isct = hit;
            }
        }
        return isct;
    }

    // Returns the incoming radiance of `Ray` via unidirectional path tracing
    float3 trace_ray(const Ray & ray, UniformRandomGenerator & gen, float3 weight, const int depth)
    {
        // Early exit with no radiance
        if (depth >= maxRecursion || luminance(weight) <= 0.0f) return float3(0, 0, 0);

        RayIntersection intersection = scene_intersects(ray);

        // Assuming no intersection, early exit with the environment color
        if (!intersection())
        {
            return environment;
        }

        BSDF * bsdf = intersection.m;

        // Russian roulette termination
        const float p = gen.random_float_safe(); // In the range [0.001f, 0.999f)
        float shouldContinue = std::min(luminance(weight), 1.f);
        if (p > shouldContinue) return float3(0.f, 0.f, 0.f);
        else weight /= shouldContinue;

        float3 tangent;
        float3 bitangent;
        make_tangent_frame(normalize(intersection.normal), tangent, bitangent);

        IntersectionInfo * surfaceInfo = new IntersectionInfo();
        surfaceInfo->Wo = -ray.direction;
        surfaceInfo->P = ray.direction * intersection.d + ray.origin;
        surfaceInfo->N = normalize(intersection.normal);
        surfaceInfo->T = normalize(tangent);
        surfaceInfo->BT = normalize(bitangent);
        surfaceInfo->Kd = bsdf->Kd;

        // Create a new BSDF event with the relevant intersection data
        SurfaceScatterEvent scatter(surfaceInfo);

        // Sample from direct light sources
        float3 directLighting;

        bool emissive = false;
        if (dynamic_cast<Emissive*>(bsdf))
        {
            emissive = true;
            directLighting += bsdf->Kd;
        }

        for (const auto light : lights)
        {
            float3 lightWi;
            float lightPDF;
            float3 lightSample = light->sample_direct(gen, surfaceInfo->P, lightWi, lightPDF);

            // Make a shadow ray to check for occlusion between surface and a direct light
            RayIntersection occlusion = scene_intersects({ surfaceInfo->P, lightWi });

            // If it's not occluded we can see the light source
            if (!occlusion())
            {
                // Sample from the BSDF
                IntersectionInfo * lightInfo = new IntersectionInfo();
                lightInfo->Wo = lightWi;
                lightInfo->P = ray.direction * intersection.d + ray.origin;
                lightInfo->N = intersection.normal;

                SurfaceScatterEvent direct(lightInfo);
                auto surfaceColor = bsdf->sample(gen, direct);

                if (direct.pdf <= 0.f || lightPDF <= 0.f) break;

                // Integrate over the number of direct lighting samples
                float3 Ld;
                for (int i = 0; i < light->numSamples; ++i)
                {
                    Ld += lightSample * surfaceColor;
                }

                directLighting += (Ld / float3(light->numSamples)) / lightPDF;

                delete lightInfo;
            }
        }

        // Sample the diffuse brdf of the intersected material
        float3 brdfSample = bsdf->sample(gen, scatter);

        // To global
        float3 sampleDirection = scatter.Wi;

        if (scatter.pdf <= 0.f || brdfSample == float3(0, 0, 0)) return float3(0, 0, 0);

        const float NdotL = clamp(float(std::abs(dot(sampleDirection, scatter.info->N))), 0.f, 1.f);

        // Weight, aka throughput
        weight *= (brdfSample * NdotL) / scatter.pdf;

        // Reflected illuminance
        float3 refl;
        if (length(sampleDirection) > 0.0f)
        {
            float3 originWithEpsilon = surfaceInfo->P + (float(0.0001f) * sampleDirection);
            refl = trace_ray(Ray(originWithEpsilon, sampleDirection), gen, weight, depth + 1);
        }

        // Free the hit struct
        delete surfaceInfo;

        return clamp(weight * directLighting + refl, 0.f, 1.f);
    }
};

//////////////
//   Film   //
//////////////

struct Film
{
    std::vector<float3> samples;
    float2 size;
    Pose view = {};
    float FoV = std::tan(to_radians(90.f) * 0.5f);

    Film(const int2 & size, const Pose & view) : samples(size.x * size.y), size(size), view(view) { }

    void set_field_of_view(float degrees) { FoV = std::tan(to_radians(degrees) * 0.5f); }

    void reset(const Pose newView)
    {
        view = newView;
        std::fill(samples.begin(), samples.end(), float3(0, 0, 0));
    }

    Ray make_ray_for_coordinate(const int2 & coord, UniformRandomGenerator & gen) const
    {
        const float aspectRatio = size.x / size.y;

        // Jitter the sampling direction and apply a tent filter for anti-aliasing
        const float r1 = 2.0f * gen.random_float();
        const float dx = (r1 < 1.0f) ? (std::sqrt(r1) - 1.0f) : (1.0f - std::sqrt(2.0f - r1));
        const float r2 = 2.0f * gen.random_float();
        const float dy = (r2 < 1.0f) ? (std::sqrt(r2) - 1.0f) : (1.0f - std::sqrt(2.0f - r2));

        const float xNorm = ((size.x * 0.5f - float(coord.x) + dx) / size.x * aspectRatio) * FoV;
        const float yNorm = ((size.y * 0.5f - float(coord.y) + dy) / size.y) * FoV;
        const float3 vNorm = float3(xNorm, yNorm, 1.0f);

        return view * Ray(float3(0.f), -(vNorm));
    }

    // Records the result of a ray traced through the camera origin (view) for a given pixel coordinate
    void trace_samples(Scene & scene, UniformRandomGenerator & gen, const int2 & coord, float numSamples)
    {
        // Integrating a cosine factor about a hemisphere yields Pi. 
        // The probability density function (PDF) of a cosine-weighted hemi is 1.f / Pi,
        // resulting in a final weight of 1.f / numSamples for Monte-Carlo integration.
        const float invSamples = 1.f / numSamples;

        float3 radiance;
        for (int s = 0; s < numSamples; ++s)
        {
            radiance = radiance + scene.trace_ray(make_ray_for_coordinate(coord, gen), gen, float3(1.f), 0);
        }

        samples[coord.y * size.x + coord.x] = radiance * invSamples;
    }

    float3 debug_trace(Scene & scene, UniformRandomGenerator & gen, const int2 & coord, float numSamples)
    {
        const float invSamples = 1.f / numSamples;

        float3 radiance;
        for (int s = 0; s < numSamples; ++s)
        {
            radiance = radiance + scene.trace_ray(make_ray_for_coordinate(coord, gen), gen, float3(1.f), 0);
        }
        return radiance *= invSamples;
    }

};

struct RendererState
{
    UniformRandomGenerator gen;
    Scene scene;
    std::vector<int2> coordinates;
    int samplesPerPixel = 32;
    simple_camera camera = {};
    std::shared_ptr<Film> film;
};

// This is essentially a quickly implemented, non-generic threadpool. 
struct ThreadedRenderCoordinator
{
    std::mutex coordinateLock;
    std::atomic<bool> earlyExit = { false };

    std::map<std::thread::id, manual_timer> renderTimers;
    std::mutex rLock;
    std::condition_variable renderCv;

    std::vector<std::thread> renderWorkers;
    std::map<std::thread::id, std::atomic<bool>> threadTaskState;
    std::atomic<int> numIdleThreads;

    const int numWorkers = std::thread::hardware_concurrency();

    std::shared_ptr<RendererState> state;

    ThreadedRenderCoordinator(std::shared_ptr<RendererState> s) : state(s)
    {
        numIdleThreads.store(0);

        for (int i = 0; i < numWorkers; ++i)
        {
            renderWorkers.push_back(std::thread(&ThreadedRenderCoordinator::threaded_render, this, generate_bag_of_pixels()));
        }
    }

    ~ThreadedRenderCoordinator()
    {
        earlyExit = true;
        for (auto & ts : threadTaskState)
        {
            ts.second.store(true);
        }

        renderCv.notify_all(); // notify render threads to wake up

        std::for_each(renderWorkers.begin(), renderWorkers.end(), [](std::thread & t)
        {
            if (t.joinable())
            {
                t.join();
            }
        });
    }

    void threaded_render(std::vector<int2> pixelCoords)
    {
        auto & timer = renderTimers[std::this_thread::get_id()];
        threadTaskState[std::this_thread::get_id()].store(false);

        while (earlyExit == false)
        {
            for (auto coord : pixelCoords)
            {
                timer.start();
                state->film->trace_samples(state->scene, state->gen, coord, state->samplesPerPixel);
                timer.stop();
            }
            pixelCoords = generate_bag_of_pixels();

            if (pixelCoords.size() == 0)
            {
                std::unique_lock<std::mutex> l(rLock);
                numIdleThreads++;
                renderCv.wait(l, [this]() { return threadTaskState[std::this_thread::get_id()].load(); });
                threadTaskState[std::this_thread::get_id()].store(false);
                numIdleThreads--;
            }
        }
    }

    // Return a vector of 1024 randomly selected coordinates from the total that we need to render.
    std::vector<int2> generate_bag_of_pixels()
    {
        std::lock_guard<std::mutex> guard(coordinateLock);
        std::vector<int2> group;
        for (int w = 0; w < 1024; w++)
        {
            if (state->coordinates.size())
            {
                auto randomIdx = state->gen.random_int((int)state->coordinates.size() - 1);
                auto randomCoord = state->coordinates[randomIdx];
                state->coordinates.erase(state->coordinates.begin() + randomIdx);
                group.push_back(randomCoord);
            }
        }

        return group;
    }

    void reset()
    {
        std::lock_guard<std::mutex> guard(coordinateLock);
        state->coordinates.clear();
        for (int y = 0; y < state->film->size.y; ++y)
        {
            for (int x = 0; x < state->film->size.x; ++x)
            {
                state->coordinates.push_back(int2(x, y));
            }
        }

        state->film->reset(state->camera.get_pose());
        // todo - take screenshot

        {
            rLock.lock();
            for (auto & ts : threadTaskState)
            {
                ts.second.store(true);
            }
            renderCv.notify_all(); // notify all threads to wake up
            rLock.unlock();
        }
    }

    bool is_idle() const { return numIdleThreads == numWorkers; }
};

void draw_texture_buffer(float rx, float ry, float rw, float rh, const GLuint handle)
{
    glBindTexture(GL_TEXTURE_2D, handle);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(rx, ry);
    glTexCoord2f(1, 0); glVertex2f(rx + rw, ry);
    glTexCoord2f(1, 1); glVertex2f(rx + rw, ry + rh);
    glTexCoord2f(0, 1); glVertex2f(rx, ry + rh);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
}

//////////////////////////
//   Main Application   //
//////////////////////////

#define WIDTH int(640)
#define HEIGHT int(480)

struct FrameInputState
{
    float2 lastCursor;
    bool ml = 0, mr = 0, bf = 0, bl = 0, bb = 0, br = 0;
};

static bool g_debug = false;
static bool takeScreenshot = true;

FrameInputState inputState;

std::unique_ptr<Window> win;
std::unique_ptr<GlShader> flatShader;
std::unique_ptr<ImGuiManager> imgui;
GlTexture2D surface;

float fieldOfView = 90;
size_t frameCount = 0;

std::shared_ptr<ThreadedRenderCoordinator> coordinator;
std::shared_ptr<RendererState> rendererState;

SimpleTimer sceneTimer;

int main(int argc, char * argv[])
{
    // Setup Window
    try
    {
        win.reset(new Window(WIDTH * 2, HEIGHT, "CPU Light Transport"));
    }
    catch (const std::exception & e)
    {
        std::cout << "Caught GLFW window exception: " << e.what() << std::endl;
    }

    // Setup application callbacks
    win->on_char = [&](int codepoint)
    {
        auto e = make_input_event(win->get_glfw_window_handle(), InputEvent::CHAR, win->get_cursor_pos(), 0);
        e.value[0] = codepoint;
        if (win->on_input) win->on_input(e);
    };

    win->on_key = [&](int key, int action, int mods)
    {
        auto e = make_input_event(win->get_glfw_window_handle(), InputEvent::KEY, win->get_cursor_pos(), action);
        e.value[0] = key;
        if (win->on_input) win->on_input(e);
    };

    win->on_mouse_button = [&](int button, int action, int mods)
    {
        auto e = make_input_event(win->get_glfw_window_handle(), InputEvent::MOUSE, win->get_cursor_pos(), action);
        e.value[0] = button;
        if (win->on_input) win->on_input(e);
    };

    win->on_cursor_pos = [&](linalg::aliases::float2 position)
    {
        auto e = make_input_event(win->get_glfw_window_handle(), InputEvent::CURSOR, position, 0);
        if (win->on_input) win->on_input(e);
    };

    win->on_input = [&](const InputEvent & event)
    {
        imgui->update_input(event);

        if (event.type == InputEvent::KEY)
        {
            if (event.value[0] == GLFW_KEY_W) inputState.bf = event.is_down();
            if (event.value[0] == GLFW_KEY_A) inputState.bl = event.is_down();
            if (event.value[0] == GLFW_KEY_S) inputState.bb = event.is_down();
            if (event.value[0] == GLFW_KEY_D) inputState.br = event.is_down();
            if (event.action == GLFW_RELEASE)
            {
                if (rendererState->camera.get_pose() != rendererState->film->view)
                {
                    coordinator->reset();
                }
            }
        }
        else if (event.type == InputEvent::MOUSE)
        {
            if (event.value[0] == GLFW_MOUSE_BUTTON_LEFT) inputState.ml = event.is_down();
            if (event.value[0] == GLFW_MOUSE_BUTTON_RIGHT) inputState.mr = event.is_down();
            if (event.action == GLFW_RELEASE)
            {
                g_debug = true;
                auto sample = rendererState->film->debug_trace(rendererState->scene, rendererState->gen, int2(event.cursor.x, HEIGHT - event.cursor.y), 1); // note 4 instead of samplesPerPixel
                std::cout << "Debug Trace: " << sample << std::endl;
                g_debug = false;
            }
        }
        else if (event.type == InputEvent::CURSOR)
        {
            auto deltaCursorMotion = event.cursor - inputState.lastCursor;
            if (inputState.mr)
            {
                rendererState->camera.yaw -= deltaCursorMotion.x * 0.01f;
                rendererState->camera.pitch -= deltaCursorMotion.y * 0.01f;
            }
            inputState.lastCursor = float2(event.cursor.x, event.cursor.y);
        }
    };

    // Create imgui context
    imgui.reset(new ImGuiManager(win->get_glfw_window_handle()));
    gui::make_dark_theme();

    // Setup path tracer
    rendererState.reset(new RendererState());

    rendererState->camera.yfov = 1.0f;
    rendererState->camera.near_clip = 0.01f;
    rendererState->camera.far_clip = 32.0f;
    rendererState->camera.position = { 0, +1.25f, 4.5f };

    rendererState->film = std::make_shared<Film>(int2(WIDTH, HEIGHT), rendererState->camera.get_pose());
    rendererState->scene.ambient = float3(0.0f);
    rendererState->scene.environment = float3(0.0f);

    std::shared_ptr<RaytracedQuad> q = std::make_shared<RaytracedQuad>();

    std::shared_ptr<RaytracedSphere> a = std::make_shared<RaytracedSphere>();
    std::shared_ptr<RaytracedSphere> b = std::make_shared<RaytracedSphere>();
    std::shared_ptr<RaytracedSphere> c = std::make_shared<RaytracedSphere>();
    std::shared_ptr<RaytracedSphere> d = std::make_shared<RaytracedSphere>();
    std::shared_ptr<RaytracedSphere> glassSphere = std::make_shared<RaytracedSphere>();

    std::shared_ptr<RaytracedBox> floor = std::make_shared<RaytracedBox>();
    std::shared_ptr<RaytracedBox> leftWall = std::make_shared<RaytracedBox>();
    std::shared_ptr<RaytracedBox> rightWall = std::make_shared<RaytracedBox>();
    std::shared_ptr<RaytracedBox> backWall = std::make_shared<RaytracedBox>();

    std::shared_ptr<PointLight> pointLight = std::make_shared<PointLight>();
    pointLight->lightPos = float3(0, 2, 0);
    pointLight->intensity = float3(1, 1, 1);
    rendererState->scene.lights.push_back(pointLight);

    {
        a->m = std::make_shared<IdealDiffuse>();
        a->radius = 0.5f;
        a->m->Kd = float3(1, 1, 1);
        a->center = float3(-0.66f, 0.50f, 0);

        b->m = std::make_shared<IdealDiffuse>();
        b->radius = 0.5f;
        b->m->Kd = float3(1, 1, 1);
        b->center = float3(+0.66f, 0.50f, 0);

        c->m = std::make_shared<IdealDiffuse>();
        c->radius = 0.5f;
        c->m->Kd = float3(1, 1, 1);
        c->center = float3(-0.33f, 0.50f, +0.66f);

        d->m = std::make_shared<IdealDiffuse>();
        d->radius = 0.5f;
        d->m->Kd = float3(1, 1, 1);
        d->center = float3(+0.33f, 0.50f, 0.66f);

        glassSphere->m = std::make_shared<DialectricBSDF>();
        glassSphere->radius = 0.50f;
        glassSphere->m->Kd = float3(1.f);
        glassSphere->center = float3(1.5f, 0.5f, 1.25);

        floor->m = std::make_shared<IdealSpecular>();
        floor->m->Kd = float3(0.9, 0.9, 0.9);
        floor->_min = float3(-2.66, -0.1, -2.66);
        floor->_max = float3(+2.66, +0.0, +2.66);

        leftWall->m = std::make_shared<IdealDiffuse>();
        leftWall->m->Kd = float3(255.f / 255.f, 20.f / 255.f, 25.f / 255.f);
        leftWall->_min = float3(-2.55, 0.f, -2.66);
        leftWall->_max = float3(-2.66, 2.66f, +2.66);

        rightWall->m = std::make_shared<IdealDiffuse>();
        rightWall->m->Kd = float3(25.f / 255.f, 255.f / 255.f, 20.f / 255.f);
        rightWall->_min = float3(+2.66, 0.f, -2.66);
        rightWall->_max = float3(+2.55, 2.66f, +2.66);

        backWall->m = std::make_shared<IdealDiffuse>();
        backWall->m->Kd = float3(0.9, 0.9, 0.9);
        backWall->_min = float3(-2.66, 0.0f, -2.66);
        backWall->_max = float3(+2.66, +2.66f, -2.55);

        rendererState->scene.objects.push_back(floor);
        rendererState->scene.objects.push_back(leftWall);
        rendererState->scene.objects.push_back(rightWall);
        rendererState->scene.objects.push_back(backWall);

        //scene.objects.push_back(a);
        //scene.objects.push_back(b);
        //scene.objects.push_back(c);
        rendererState->scene.objects.push_back(d);
        rendererState->scene.objects.push_back(glassSphere);
        rendererState->scene.objects.push_back(q);

        q->q.reset(new Quad(float3(-0.5, 0, 0), float3(0, 1, 0), float3(1, 0, 0)));
        q->m = std::make_shared<Emissive>();
        q->m->Kd = float3(1, 0, 0);

        std::shared_ptr<AreaLight> areaLight = std::make_shared<AreaLight>();
        areaLight->intensity = float3(1, 1, 1);
        areaLight->quad = q->q.get();
        rendererState->scene.lights.push_back(areaLight);

        // Traverse + build BVH accelerator for the objects we've added to the scene
        {
            scoped_timer bvh("BVH Generation");
            //scene.accelerate();
        }

        // Generate a vector of all possible pixel locations to raytrace
        for (int y = 0; y < rendererState->film->size.y; ++y)
        {
            for (int x = 0; x < rendererState->film->size.x; ++x)
            {
                rendererState->coordinates.push_back(int2(x, y));
            }
        }

        coordinator.reset(new ThreadedRenderCoordinator(rendererState));

        surface.setup(WIDTH, HEIGHT, GL_RGB, GL_RGB, GL_FLOAT, nullptr);
    }

    int2 windowSize = win->get_window_size();

    sceneTimer.start();

    // Application main loop
    auto t0 = std::chrono::high_resolution_clock::now();
    while (!win->should_close())
    {
        glfwMakeContextCurrent(win->get_glfw_window_handle());

        glfwPollEvents();

        auto t1 = std::chrono::high_resolution_clock::now();
        float timestep = std::chrono::duration<float>(t1 - t0).count();
        t0 = t1;

        if (inputState.mr)
        {
            const linalg::aliases::float4 orientation = rendererState->camera.get_orientation();
            linalg::aliases::float3 move;
            if (inputState.bf) move -= qzdir(orientation);
            if (inputState.bl) move -= qxdir(orientation);
            if (inputState.bb) move += qzdir(orientation);
            if (inputState.br) move += qxdir(orientation);
            if (length2(move) > 0)  rendererState->camera.position += normalize(move) * (timestep * 10);
        }

        // Upload path-traced data to a gpu surface for realtime visualization
        {
            auto pixelsCopy = rendererState->film->samples;
            int width = rendererState->film->size.x;
            int height = rendererState->film->size.y;
            flip_vertical_inplace(pixelsCopy, int2(width, height));
            glTextureImage2DEXT(surface, GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_FLOAT, pixelsCopy.data());
        }

        glPushMatrix();
        glOrtho(0, windowSize.x, windowSize.y, 0, -1, +1);

        glUseProgram(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        draw_texture_buffer(0.f, 0.f, (float)WIDTH, float(HEIGHT), surface);
        glPopMatrix();

        gl_check_error(__FILE__, __LINE__);

        // ImGui
        {
            imgui->begin_frame();

            ImGui::Text("Application Runtime %.3lld seconds", sceneTimer.seconds().count());
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::InputFloat3("Camera Position", &rendererState->camera.get_pose().position[0]);
            ImGui::InputFloat4("Camera Orientation", &rendererState->camera.get_pose().orientation[0]);
            ImGui::ColorEdit3("Ambient Color", &rendererState->scene.ambient[0]);

            if (ImGui::SliderFloat("Camera FoV", &fieldOfView, 45.f, 120.f))
            {
                coordinator->reset();
                rendererState->film->set_field_of_view(fieldOfView);
            }

            if (ImGui::SliderInt("SPP", &rendererState->samplesPerPixel, 1, 8192))
            {
                coordinator->reset();
            }

            for (auto & t : coordinator->renderTimers)
            {
                ImGui::Text("%#010x %.3f", t.first, t.second.get());
            }

            if (ImGui::Button("Save *.png")) take_screenshot({ WIDTH, HEIGHT });

            imgui->end_frame();
        }

        frameCount++;

        // Have we finished rendering? 
        if (coordinator->is_idle() && takeScreenshot == true)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            takeScreenshot = take_screenshot({ WIDTH, HEIGHT });
            sceneTimer.stop();
            std::cout << "Render Saved..." << std::endl;
        }

        win->swap_buffers();
    }
}