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
#include <random>
#include <ctime>

#include "linalg_util.hpp"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "gl-api.hpp"
#include "geometric.hpp"

#define PI            3.1415926535897931
#define HALF_PI       1.5707963267948966
#define QUARTER_PI    0.7853981633974483
#define TWO_PI        6.2831853071795862
#define TAU           TWO_PI
#define INV_PI        0.3183098861837907
#define INV_TWO_PI    0.1591549430918953
#define INV_HALF_PI   0.6366197723675813

template<typename T>
T clamp(const T & val, const T & min, const T & max)
{
    return std::min(std::max(val, min), max);
}

inline float to_radians(float degrees) { return degrees * float(PI) / 180.f; }
inline float to_degrees(float radians) { return radians * 180.f / float(PI); }

class UniformRandomGenerator
{
    std::random_device rd;
    std::mt19937_64 gen;
    std::uniform_real_distribution<float> full{ 0.f, 1.f };
    std::uniform_real_distribution<float> safe{ 0.001f, 0.999f };
    std::uniform_real_distribution<float> two_pi{ 0.f, float(TWO_PI) };
public:
    UniformRandomGenerator() : rd(), gen(rd()) { }
    float random_float() { return full(gen); }
    float random_float(float max) { std::uniform_real_distribution<float> custom(0.f, max); return custom(gen); }
    float random_float_sphere() { return two_pi(gen); }
    float random_float_safe() { return safe(gen); }
    int random_int(int max) { std::uniform_int_distribution<int> dInt(0, max); return dInt(gen); }
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

class SimpleTimer
{
    typedef std::chrono::high_resolution_clock::time_point timepoint;
    typedef std::chrono::high_resolution_clock::duration timeduration;

    bool isRunning;
    timepoint startTime;
    timepoint pauseTime;

    inline timepoint current_time_point() const { return std::chrono::high_resolution_clock::now(); }
    inline timeduration running_time() const { return (isRunning) ? current_time_point() - startTime : pauseTime - startTime; }

    template<typename unit>
    inline unit running_time() const
    {
        return std::chrono::duration_cast<unit>(running_time());
    }

public:

    SimpleTimer(bool run = false) : isRunning(run)
    {
        if (run) start();
    }

    void start()
    {
        reset();
        isRunning = true;
    }

    void stop()
    {
        reset();
        isRunning = false;
    }

    void reset()
    {
        startTime = std::chrono::high_resolution_clock::now();
        pauseTime = startTime;
    }

    void pause()
    {
        pauseTime = current_time_point();
        isRunning = false;
    }

    void unpause()
    {
        if (isRunning) return;
        startTime += current_time_point() - pauseTime;
        isRunning = true;
    }

    std::chrono::nanoseconds nanoseconds() const { return running_time<std::chrono::nanoseconds>(); }
    std::chrono::microseconds microseconds() const { return running_time<std::chrono::microseconds>(); }
    std::chrono::milliseconds milliseconds() const { return running_time<std::chrono::milliseconds>(); }
    std::chrono::seconds seconds() const { return running_time<std::chrono::seconds>(); }
    bool is_running() { return isRunning; }
};

class manual_timer
{
    std::chrono::high_resolution_clock::time_point t0;
    double timestamp{ 0.f };
public:
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    void stop() { timestamp = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - t0).count() * 1000; }
    const double & get() { return timestamp; }
};

struct HumanTime
{

    typedef std::chrono::duration<int, std::ratio_multiply<std::chrono::hours::period, std::ratio<24> >::type> days;

    int year;
    int month;
    int yearDay;
    int monthDay;
    int weekDay;
    int hour;
    int minute;
    int second;
    int isDST;

    HumanTime()
    {
        update();
    }

    void update()
    {
        std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
        std::time_t tt = std::chrono::system_clock::to_time_t(now);
        tm local_tm = *localtime(&tt);
        year = local_tm.tm_year + 1900;
        month = local_tm.tm_mon + 1;
        monthDay = local_tm.tm_mday;
        yearDay = local_tm.tm_yday;
        weekDay = local_tm.tm_wday;
        hour = local_tm.tm_hour;
        minute = local_tm.tm_min;
        second = local_tm.tm_sec;
        isDST = local_tm.tm_isdst;
    }

    std::string make_timestamp()
    {
        std::string timestamp =
            std::to_string(month) + "." +
            std::to_string(monthDay) + "." +
            std::to_string(year) + "-" +
            std::to_string(hour) + "." +
            std::to_string(minute) + "." +
            std::to_string(second);
        return timestamp;
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

///////////////////////////////////
//   Windowing & App Lifecycle   //
///////////////////////////////////

struct simple_camera
{
    float yfov, near_clip, far_clip;
    float3 position;
    float pitch, yaw;
    Pose get_pose() const { return{ get_orientation(), position }; }
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

struct screen_viewport
{
    float2 bmin, bmax;
    GLuint texture;
};

#endif // end common_utils_hpp
