// This is free and unencumbered software released into the public domain.
// For more information, please refer to <http://unlicense.org>

#pragma once

#ifndef imgui_gl_wrapper_hpp
#define imgui_gl_wrapper_hpp

#include <memory>
#include <vector>
#include <map>
#include <string>

#include "linalg_util.hpp"
#include "util.hpp"

// Implicit casts for linalg types
#define IM_VEC2_CLASS_EXTRA                                             \
ImVec2(const float2 & f) { x = f.x; y = f.y; }                          \
operator float2() const { return float2(x,y); }                         \
ImVec2(const int2 & f) { x = f.x; y = f.y; }                            \
operator int2() const { return int2(x,y); }

#define IM_VEC4_CLASS_EXTRA                                             \
ImVec4(const float4 & f) { x = f.x; y = f.y; z = f.z; w = f.w; }        \
operator float4() const { return float4(x,y,z,w); }

#include "imgui/imgui.h"

struct ui_rect
{
    int2 min, max;
    bool contains(const int2 & p) const { return all(gequal(p, min)) && all(less(p, max)); }
};

struct GlTexture2D;

struct GLFWwindow;
namespace gui
{

    struct ImGuiApp : public Singleton<ImGuiApp>
    {
        GLFWwindow * window;
        double       Time = 0.0f;
        bool         MousePressed[3] = { false, false, false };
        float        MouseWheel = 0.0f;
        uint32_t     FontTexture = 0;
        int          ShaderHandle = 0, VertHandle = 0, FragHandle = 0;
        int          AttribLocationTex = 0, AttribLocationProjMtx = 0;
        int          AttribLocationPosition = 0, AttribLocationUV = 0, AttribLocationColor = 0;
        unsigned int VboHandle = 0, VaoHandle = 0, ElementsHandle = 0;
        friend class Singleton<ImGuiApp>;
    };

    class ImGuiManager
    {
        bool create_fonts_texture();
        bool create_render_objects();
        void destroy_render_objects();
    public:
        
        ImGuiManager(GLFWwindow * win);
        ~ImGuiManager();
        
        void update_input(const InputEvent & e);

        void begin_frame();
        void end_frame();

        bool capturedKeys[1024];
    };
    
    inline void make_dark_theme()
    {
        ImGuiStyle & s = ImGui::GetStyle();
        
        s.WindowMinSize                          = ImVec2(160, 20);
        s.FramePadding                           = ImVec2(4, 2);
        s.ItemSpacing                            = ImVec2(6, 2);
        s.ItemInnerSpacing                       = ImVec2(6, 4);
        s.Alpha                                  = 0.95f;
        s.WindowFillAlphaDefault                 = 1.0f;
        s.WindowRounding                         = 4.0f;
        s.FrameRounding                          = 2.0f;
        s.IndentSpacing                          = 6.0f;
        s.ColumnsMinSpacing                      = 50.0f;
        s.GrabMinSize                            = 14.0f;
        s.GrabRounding                           = 16.0f;
        s.ScrollbarSize                          = 12.0f;
        s.ScrollbarRounding                      = 16.0f;
        
        s.Colors[ImGuiCol_Text]                  = ImVec4(0.86f, 0.93f, 0.89f, 0.61f);
        s.Colors[ImGuiCol_TextDisabled]          = ImVec4(0.86f, 0.93f, 0.89f, 0.28f);
        s.Colors[ImGuiCol_WindowBg]              = ImVec4(0.13f, 0.14f, 0.17f, 1.00f);
        s.Colors[ImGuiCol_ChildWindowBg]         = ImVec4(0.20f, 0.22f, 0.27f, 0.58f);
        s.Colors[ImGuiCol_Border]                = ImVec4(0.31f, 0.31f, 1.00f, 0.00f);
        s.Colors[ImGuiCol_BorderShadow]          = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        s.Colors[ImGuiCol_FrameBg]               = ImVec4(0.20f, 0.22f, 0.27f, 1.00f);
        s.Colors[ImGuiCol_FrameBgHovered]        = ImVec4(0.92f, 0.18f, 0.29f, 0.78f);
        s.Colors[ImGuiCol_FrameBgActive]         = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        s.Colors[ImGuiCol_TitleBg]               = ImVec4(0.20f, 0.22f, 0.27f, 1.00f);
        s.Colors[ImGuiCol_TitleBgCollapsed]      = ImVec4(0.20f, 0.22f, 0.27f, 0.75f);
        s.Colors[ImGuiCol_TitleBgActive]         = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        s.Colors[ImGuiCol_MenuBarBg]             = ImVec4(0.20f, 0.22f, 0.27f, 0.47f);
        s.Colors[ImGuiCol_ScrollbarBg]           = ImVec4(0.20f, 0.22f, 0.27f, 1.00f);
        s.Colors[ImGuiCol_ScrollbarGrab]         = ImVec4(0.47f, 0.77f, 0.83f, 0.21f);
        s.Colors[ImGuiCol_ScrollbarGrabHovered]  = ImVec4(0.92f, 0.18f, 0.29f, 0.78f);
        s.Colors[ImGuiCol_ScrollbarGrabActive]   = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        s.Colors[ImGuiCol_ComboBg]               = ImVec4(0.20f, 0.22f, 0.27f, 1.00f);
        s.Colors[ImGuiCol_CheckMark]             = ImVec4(0.71f, 0.22f, 0.27f, 1.00f);
        s.Colors[ImGuiCol_SliderGrab]            = ImVec4(0.47f, 0.77f, 0.83f, 0.14f);
        s.Colors[ImGuiCol_SliderGrabActive]      = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        s.Colors[ImGuiCol_Button]                = ImVec4(0.47f, 0.77f, 0.83f, 0.14f);
        s.Colors[ImGuiCol_ButtonHovered]         = ImVec4(0.92f, 0.18f, 0.29f, 0.86f);
        s.Colors[ImGuiCol_ButtonActive]          = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        s.Colors[ImGuiCol_Header]                = ImVec4(0.92f, 0.18f, 0.29f, 0.76f);
        s.Colors[ImGuiCol_HeaderHovered]         = ImVec4(0.92f, 0.18f, 0.29f, 0.86f);
        s.Colors[ImGuiCol_HeaderActive]          = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        s.Colors[ImGuiCol_Column]                = ImVec4(0.47f, 0.77f, 0.83f, 0.32f);
        s.Colors[ImGuiCol_ColumnHovered]         = ImVec4(0.92f, 0.18f, 0.29f, 0.78f);
        s.Colors[ImGuiCol_ColumnActive]          = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        s.Colors[ImGuiCol_ResizeGrip]            = ImVec4(0.47f, 0.77f, 0.83f, 0.04f);
        s.Colors[ImGuiCol_ResizeGripHovered]     = ImVec4(0.92f, 0.18f, 0.29f, 0.78f);
        s.Colors[ImGuiCol_ResizeGripActive]      = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        s.Colors[ImGuiCol_CloseButton]           = ImVec4(0.86f, 0.93f, 0.89f, 0.16f);
        s.Colors[ImGuiCol_CloseButtonHovered]    = ImVec4(0.86f, 0.93f, 0.89f, 0.39f);
        s.Colors[ImGuiCol_CloseButtonActive]     = ImVec4(0.86f, 0.93f, 0.89f, 1.00f);
        s.Colors[ImGuiCol_PlotLines]             = ImVec4(0.86f, 0.93f, 0.89f, 0.63f);
        s.Colors[ImGuiCol_PlotLinesHovered]      = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        s.Colors[ImGuiCol_PlotHistogram]         = ImVec4(0.86f, 0.93f, 0.89f, 0.63f);
        s.Colors[ImGuiCol_PlotHistogramHovered]  = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
        s.Colors[ImGuiCol_TextSelectedBg]        = ImVec4(0.92f, 0.18f, 0.29f, 0.43f);
        s.Colors[ImGuiCol_TooltipBg]             = ImVec4(0.47f, 0.77f, 0.83f, 0.72f);
        s.Colors[ImGuiCol_ModalWindowDarkening]  = ImVec4(0.20f, 0.22f, 0.27f, 0.73f);
    }

    inline void imgui_fixed_window_begin(const char * name, const ui_rect & r)
    {
        ImGui::SetNextWindowPos(r.min);
        ImGui::SetNextWindowSize(r.max - r.min);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(0, 0));
        bool result = ImGui::Begin(name, NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
        ImGui::TextColored({ 1,1,0.5f,1 }, name);
        ImGui::Separator();
        assert(result);
    }

    inline void imgui_fixed_window_end()
    {
        ImGui::End();
        ImGui::PopStyleVar(2);
    }
}

#endif // imgui_gl_wrapper_hpp
