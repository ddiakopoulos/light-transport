// This is free and unencumbered software released into the public domain.
// For more information, please refer to <http://unlicense.org>

#pragma once

#ifndef common_utils_hpp
#define common_utils_hpp

#include <functional>
#include <vector>
#include <chrono>
#include <string>
#include <exception>
#include <sstream>
#include <memory>

#include "linalg_util.hpp"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "gl-api.hpp"

#define PI            3.1415926535897931
#define HALF_PI       1.5707963267948966
#define QUARTER_PI    0.7853981633974483
#define TWO_PI        6.2831853071795862
#define TAU           TWO_PI
#define INV_PI        0.3183098861837907
#define INV_TWO_PI    0.1591549430918953
#define INV_HALF_PI   0.6366197723675813

template <typename T, int C>
struct image_buffer
{
    const int2 size;
    T * alias;
    struct delete_array { void operator()(T * p) { delete[] p; } };
    std::unique_ptr<T, decltype(image_buffer::delete_array())> data;
    image_buffer() : size({ 0, 0 }) { }
    image_buffer(const int2 size, T * ptr) : size(size) { alias = ptr; }
    image_buffer(const int2 size) : size(size), data(new T[size.x * size.y * C], delete_array()) { alias = data.get(); }
    image_buffer(const image_buffer<T, C> & r) : size(r.size), data(new T[size.x * size.y * C], delete_array())
    {
        alias = data.get();
        if (r.alias) std::memcpy(alias, r.alias, size.x * size.y * C * sizeof(T));
    }
    int size_bytes() const { return C * size.x * size.y * sizeof(T); }
    int num_pixels() const { return size.x * size.y; }
    T & operator()(int y, int x) { return alias[y * size.x + x]; }
    T & operator()(int y, int x, int channel) { return alias[C * (y * size.x + x) + channel]; }
};

class scoped_timer
{
    std::string message;
    std::chrono::high_resolution_clock::time_point t0;
public:
    scoped_timer(std::string message) : message{ std::move(message) }, t0{ std::chrono::high_resolution_clock::now() } {}
    ~scoped_timer()
    {
        std::cout << message << " completed in " << std::to_string((std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - t0).count() * 1000)) << " ms" << std::endl;
    }
};

class Noncopyable
{
protected:
    Noncopyable() = default;
    ~Noncopyable() = default;
    Noncopyable(const Noncopyable& r) = delete;
    Noncopyable & operator = (const Noncopyable& r) = delete;
};

template <typename T>
class Singleton : public Noncopyable
{
private:
    Singleton(const Singleton<T> &);
    Singleton & operator = (const Singleton<T> &);
protected:
    static T * single;
    Singleton() = default;
    ~Singleton() = default;
public:
    static T & get_instance() { if (!single) single = new T(); return *single; };
};

inline std::vector<uint8_t> read_file_binary(const std::string pathToFile)
{
    FILE * f = fopen(pathToFile.c_str(), "rb");

    if (!f) throw std::runtime_error("file not found");

    fseek(f, 0, SEEK_END);
    size_t lengthInBytes = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::vector<uint8_t> fileBuffer(lengthInBytes);

    size_t elementsRead = fread(fileBuffer.data(), 1, lengthInBytes, f);

    if (elementsRead == 0 || fileBuffer.size() < 4) throw std::runtime_error("error reading file or file too small");

    fclose(f);
    return fileBuffer;
}

template <typename T> uint16_t normalize(T value)
{
    const float min = std::numeric_limits<T>::min();
    const float max = std::numeric_limits<T>::max();
    const float result = (value - min) / (max - min);

    float new_float = result * std::numeric_limits<uint16_t>::max();

    if (new_float > std::numeric_limits<uint16_t>::max()) new_float = std::numeric_limits<uint16_t>::max();
    if (new_float < 0) new_float = 0;

    float round_float = std::nearbyintf(new_float);
    uint16_t round_int = static_cast<uint16_t>(round_float);

    return round_int;
}

inline float to_luminance(float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }

inline image_buffer<uint16_t, 1> rgb_to_greyscale(const int width, const int height, const int nBytes, const uint8_t * rawBytes)
{
    image_buffer<uint16_t, 1> buffer({ width, height });

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; ++x)
        {
            const float r = normalize<uint8_t>(rawBytes[nBytes * (y * width + x) + 0]);
            const float g = normalize<uint8_t>(rawBytes[nBytes * (y * width + x) + 1]);
            const float b = normalize<uint8_t>(rawBytes[nBytes * (y * width + x) + 2]);
            buffer(y, x) = to_luminance(r, g, b);
        }
    }
    return buffer;
}

inline image_buffer<uint16_t, 1> rgb_to_greyscale_stb(std::vector<uint8_t> & binaryData)
{
    int width, height, nBytes;
    auto data = stbi_load_from_memory(binaryData.data(), (int)binaryData.size(), &width, &height, &nBytes, 0);
    auto buffer = rgb_to_greyscale(width, height, nBytes, data);
    stbi_image_free(data);
    return buffer;
}

template<typename T>
T clamp(const T & val, const T & min, const T & max)
{
    return std::min(std::max(val, min), max);
}

inline float to_radians(float degrees) { return degrees * float(PI) / 180.f; }
inline float to_degrees(float radians) { return radians * 180.f / float(PI); }

inline float4x4 make_projection_matrix(float l, float r, float b, float t, float n, float f)
{
    return{ { 2 * n / (r - l),0,0,0 },{ 0,2 * n / (t - b),0,0 },{ (r + l) / (r - l),(t + b) / (t - b),-(f + n) / (f - n),-1 },{ 0,0,-2 * f*n / (f - n),0 } };
}

inline float4x4 make_perspective_matrix(float vFovInRadians, float aspectRatio, float nearZ, float farZ)
{
    const float top = nearZ * std::tan(vFovInRadians / 2.f), right = top * aspectRatio;
    return make_projection_matrix(-right, right, -top, top, nearZ, farZ);
}

inline float4x4 make_rigid_transformation_matrix(const float4 & rotation, const float3 & translation)
{
    return{ { qxdir(rotation),0 },{ qydir(rotation),0 },{ qzdir(rotation),0 },{ translation,1 } };
}

// Rigid transformation value-type
struct Pose
{
    float4      orientation;        // Orientation of an object, expressed as a rotation quaternion from the base orientation
    float3      position;           // Position of an object, expressed as a translation vector from the base position

    Pose() : Pose({ 0,0,0,1 }, { 0,0,0 }) {}
    Pose(const float4 & orientation, const float3 & position) : orientation(orientation), position(position) {}
    explicit    Pose(const float4 & orientation) : Pose(orientation, { 0,0,0 }) {}
    explicit    Pose(const float3 & position) : Pose({ 0,0,0,1 }, position) {}

    Pose        inverse() const { auto invOri = qinv(orientation); return{ invOri, qrot(invOri, -position) }; }
    float4x4    matrix() const { return make_rigid_transformation_matrix(orientation, position); }
    float3      xdir() const { return qxdir(orientation); } // Equivalent to transform_vector({1,0,0})
    float3      ydir() const { return qydir(orientation); } // Equivalent to transform_vector({0,1,0})
    float3      zdir() const { return qzdir(orientation); } // Equivalent to transform_vector({0,0,1})

    float3      transform_vector(const float3 & vec) const { return qrot(orientation, vec); }
    float3      transform_coord(const float3 & coord) const { return position + transform_vector(coord); }
    float3      detransform_coord(const float3 & coord) const { return detransform_vector(coord - position); } // Equivalent to inverse().transform_coord(coord), but faster
    float3      detransform_vector(const float3 & vec) const { return qrot(qinv(orientation), vec); } // Equivalent to inverse().transform_vector(vec), but faster

    Pose        operator * (const Pose & pose) const { return{ qmul(orientation,pose.orientation), transform_coord(pose.position) }; }
};

inline bool operator == (const Pose & a, const Pose & b)
{
    return (a.position == b.position) && (a.orientation == b.orientation);
}

inline bool operator != (const Pose & a, const Pose & b)
{
    return (a.position != b.position) || (a.orientation != b.orientation);
}

inline std::ostream & operator << (std::ostream & o, const Pose & r)
{
    return o << "{" << r.position << ", " << r.orientation << "}";
}

///////////////////////////////////
//   Windowing & App Lifecycle   //
///////////////////////////////////

struct simple_camera
{
    float yfov, near_clip, far_clip;
    float3 position;
    float pitch, yaw;
    float4 get_orientation() const { return qmul(rotation_quat(float3(0, 1, 0), yaw), rotation_quat(float3(1, 0, 0), pitch)); }
    float4x4 get_view_matrix() const { return mul(rotation_matrix(qconj(get_orientation())), translation_matrix(-position)); }
    float4x4 get_projection_matrix(const float aspectRatio) const { return linalg::perspective_matrix(yfov, aspectRatio, near_clip, far_clip); }
    float4x4 get_viewproj_matrix(const float aspectRatio) const { return mul(get_projection_matrix(aspectRatio), get_view_matrix()); }
};

struct InputEvent
{
    enum Type { CURSOR, MOUSE, KEY, CHAR, SCROLL };

    GLFWwindow * window;
    linalg::aliases::int2 windowSize;

    Type type;
    int action;
    int mods;

    linalg::aliases::float2 cursor;
    bool drag = false;

    linalg::aliases::uint2 value; // button, key, codepoint, scrollX, scrollY

    bool is_down() const { return action != GLFW_RELEASE; }
    bool is_up() const { return action == GLFW_RELEASE; }

    bool using_shift_key() const { return mods & GLFW_MOD_SHIFT; };
    bool using_control_key() const { return mods & GLFW_MOD_CONTROL; };
    bool using_alt_key() const { return mods & GLFW_MOD_ALT; };
    bool using_super_key() const { return mods & GLFW_MOD_SUPER; };
};

static InputEvent make_input_event(GLFWwindow * window, InputEvent::Type type, const linalg::aliases::float2 cursor, int action)
{
    static bool isDragging = false;

    InputEvent e;
    e.window = window;
    e.type = type;
    e.cursor = cursor;
    e.action = action;
    e.mods = 0;

    glfwGetWindowSize(window, &e.windowSize.x, &e.windowSize.y);

    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) | glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)) e.mods |= GLFW_MOD_SHIFT;
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) | glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL)) e.mods |= GLFW_MOD_CONTROL;
    if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) | glfwGetKey(window, GLFW_KEY_RIGHT_ALT)) e.mods |= GLFW_MOD_ALT;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SUPER) | glfwGetKey(window, GLFW_KEY_RIGHT_SUPER)) e.mods |= GLFW_MOD_SUPER;

    if (type == InputEvent::MOUSE)
    {
        if (e.is_down()) isDragging = true;
        else if (e.is_up()) isDragging = false;
    }
    e.drag = isDragging;

    return e;
}

class Window
{
    GLFWwindow * window;
public:
    std::function<void(unsigned int codepoint)> on_char;
    std::function<void(int key, int action, int mods)> on_key;
    std::function<void(int button, int action, int mods)> on_mouse_button;
    std::function<void(float2 pos)> on_cursor_pos;
    std::function<void(int numFiles, const char ** paths)> on_drop;
    std::function<void(InputEvent e)> on_input;

    Window(int width, int height, const char * title)
    {
        if (glfwInit() == GL_FALSE) throw std::runtime_error("glfwInit() failed");

        window = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!window) throw std::runtime_error("glfwCreateWindow() failed");

        glfwMakeContextCurrent(window);

        if (GLenum err = glewInit())
        {
            throw std::runtime_error(std::string("glewInit() failed - ") + (const char *)glewGetErrorString(err));
        }

		std::cout << "GL_VERSION =  " << (char *) glGetString(GL_VERSION) << std::endl;
		std::cout << "GL_VENDOR =   " << (char *) glGetString(GL_VENDOR) << std::endl;
		std::cout << "GL_RENDERER = " << (char *) glGetString(GL_RENDERER) << std::endl;

		check_extensions({ "GL_EXT_direct_state_access" });

        glfwSetCharCallback(window, [](GLFWwindow * window, unsigned int codepoint) {
            auto w = (Window *)glfwGetWindowUserPointer(window); if (w->on_char) w->on_char(codepoint);
        });

        glfwSetKeyCallback(window, [](GLFWwindow * window, int key, int, int action, int mods) {
            auto w = (Window *)glfwGetWindowUserPointer(window); if (w->on_key) w->on_key(key, action, mods);
        });

        glfwSetMouseButtonCallback(window, [](GLFWwindow * window, int button, int action, int mods) {
            auto w = (Window *)glfwGetWindowUserPointer(window); if (w->on_mouse_button) w->on_mouse_button(button, action, mods);
        });

        glfwSetCursorPosCallback(window, [](GLFWwindow * window, double xpos, double ypos) {
            auto w = (Window *)glfwGetWindowUserPointer(window); if (w->on_cursor_pos) w->on_cursor_pos(float2(double2(xpos, ypos)));
        });

        glfwSetDropCallback(window, [](GLFWwindow * window, int numFiles, const char ** paths) {
            auto w = (Window *)glfwGetWindowUserPointer(window); if (w->on_drop) w->on_drop(numFiles, paths);
        });

        glfwSetWindowUserPointer(window, this);
    }

    ~Window()
    {
        glfwMakeContextCurrent(window);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    Window(const Window &) = delete;
    Window(Window &&) = delete;
    Window & operator = (const Window &) = delete;
    Window & operator = (Window &&) = delete;

    GLFWwindow * get_glfw_window_handle() { return window; };
    bool should_close() const { return !!glfwWindowShouldClose(window); }
    int get_window_attrib(int attrib) const { return glfwGetWindowAttrib(window, attrib); }
    int2 get_window_size() const { int2 size; glfwGetWindowSize(window, &size.x, &size.y); return size; }
    void set_window_size(int2 newSize) { glfwSetWindowSize(window, newSize.x, newSize.y); }
    int2 get_framebuffer_size() const { int2 size; glfwGetFramebufferSize(window, &size.x, &size.y); return size; }
    float2 get_cursor_pos() const { double2 pos; glfwGetCursorPos(window, &pos.x, &pos.y); return float2(pos); }

    void swap_buffers() { glfwSwapBuffers(window); }
    void close() { glfwSetWindowShouldClose(window, 1); }
};

struct Bounds2D
{
    float2 _min = { 0, 0 };
    float2 _max = { 0, 0 };

    Bounds2D() {}
    Bounds2D(float2 min, float2 max) : _min(min), _max(max) {}
    Bounds2D(float x0, float y0, float x1, float y1) { _min.x = x0; _min.y = y0; _max.x = x1; _max.y = y1; }

    float2 min() const { return _min; }
    float2 max() const { return _max; }

    float2 size() const { return max() - min(); }
    float2 center() const { return{ (_min.x + _max.y) / 2, (_min.y + _max.y) / 2 }; }
    float area() const { return (_max.x - _min.x) * (_max.y - _min.y); }

    float width() const { return _max.x - _min.x; }
    float height() const { return _max.y - _min.y; }

    bool contains(const float px, const float py) const { return px >= _min.x && py >= _min.y && px < _max.x && py < _max.y; }
    bool contains(const float2 & point) const { return contains(point.x, point.y); }

    bool intersects(const Bounds2D & other) const
    {
        if ((_min.x <= other._min.x) && (_max.x >= other._max.x) &&
            (_min.y <= other._min.y) && (_max.y >= other._max.y)) return true;
        return false;
    }
};

struct screen_viewport
{
    float2 bmin, bmax;
    GLuint texture;
};


constexpr const char fullscreen_quad_vert[] = R"(#version 330
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec2 texcoord;
    uniform mat4 u_mvp;
    out vec2 v_texcoord;
    void main()
    {
	    gl_Position = u_mvp * vec4(position.xyz, 1);
        v_texcoord = texcoord; //(position.xy + vec2(1, 1)) / 2.0;
    }
)";

constexpr const char fullscreen_quad_frag[] = R"(#version 330
    uniform sampler2D s_diffuseTex;
    in vec2 v_texcoord;
    out vec4 f_color;
    void main()
    {
        f_color = texture(s_diffuseTex, vec2(v_texcoord.x, 1 - v_texcoord.y));
    }
)";

#endif // end common_utils_hpp
