/*
 * ImGui plugin example
 * Copyright (C) 2021 Jean Pierre Cimalando <jp-dev@inbox.ru>
 * Copyright (C) 2021-2022 Filipe Coelho <falktx@falktx.com>
 * SPDX-License-Identifier: ISC
 */

#include "DistrhoUI.hpp"
#include "ResizeHandle.hpp"
#include "Parameters.hpp"

#include <../clap/include/clap/ext/context-menu.h>
#include <../clap/include/clap/ext/state.h>
#include <../clap/include/clap/ext/params.h>
#include <windows.h>
#include <commctrl.h> // For SetWindowSubclass API
#include "PluginDSP.hpp"
#pragma comment(lib, "comctl32.lib")


START_NAMESPACE_DISTRHO


//for right click autmoaiton clip
class ImGuiPluginUI : public UI
{
    ResizeHandle fResizeHandle;
    bool needsNormalise=false;
public:
    float fA,fB,fC,fD;
    float fDelay;
    ImGuiPluginUI()
        : UI(DISTRHO_UI_DEFAULT_WIDTH,DISTRHO_UI_DEFAULT_HEIGHT),
        fResizeHandle(this)
    {

        // const double scaleFactor = getScaleFactor();

        setSize(DISTRHO_UI_DEFAULT_WIDTH,DISTRHO_UI_DEFAULT_HEIGHT);


            fResizeHandle.hide();
        // Get the global style object
        ImGuiStyle& style = ImGui::GetStyle();

        // Set the global window background color
        style.Colors[ImGuiCol_WindowBg] = ImVec4(63.0f / 255.0f, 72.0f / 255.0f, 77.0f / 255.0f, 0.12f);

    }

    void stateChanged(const char* key, const char* value){

    }

    ImGuiPluginDSP* getPluginDPSPointer(){
        auto* plugin = static_cast<ImGuiPluginDSP*>(getPluginInstancePointer());
        return plugin;
    }
    bool checkIfClapAtRuntime()
    {
        char fileBuffer[MAX_PATH] = {0};
        HMODULE hModule = NULL;

        // 🟢 Create a dummy static variable. It lives inside your plugin library's binary memory space.
        static const int dummyAnchor = 0;

        // 🟢 Pass the address of the dummy anchor variable instead of the member function pointer
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(&dummyAnchor), &hModule);

        if (hModule) {
            GetModuleFileNameA(hModule, fileBuffer, sizeof(fileBuffer));
            std::string binaryPath(fileBuffer);

            std::transform(binaryPath.begin(), binaryPath.end(), binaryPath.begin(), ::tolower);

            std::string target = ".clap";
            if (binaryPath.length() >= target.length()) {
                return (binaryPath.compare(binaryPath.length() - target.length(), target.length(), target) == 0);
            }
        }
        return false; // 🔵 Fallback (VST3, etc.)
    }

    int getPluginFormat()
    {   if(checkIfClapAtRuntime())        return 1;
        return 0;
    }

    void setDirty(){
        //only works in bitwig
        const uint32_t activeFormat = getPluginFormat();

        if (activeFormat == 1)
        {
            auto* clapPointer=reinterpret_cast<const clap_host_t*>(getPluginDPSPointer()->host);
            if(!clapPointer)return;
            auto* hostState = reinterpret_cast<const clap_host_state_t*>(clapPointer->get_extension(clapPointer, CLAP_EXT_STATE));
            if (hostState != nullptr && hostState->mark_dirty != nullptr) {
                setParameterValue(kParamDelay,(float)std::max(0,std::min(MAX_DELAY,(int)fDelay)));


                hostState->mark_dirty(clapPointer);
            }
        }
    }

    // Window& getWindow() const override {
    //     return UI::getWindow();
    // }



protected:
    void parameterChanged(uint32_t index, float value) override {
        if(index==kParamA){
            fA = value;
        }
        if(index==kParamB){
            fB=value;
        }
        if(index==kParamC){
            fC=value;
        }
        if(index==kParamD){
            fD=value;
        }
        if(index==kParamDelay){
            fDelay=value;
        }
        repaint();
    }



    void onImGuiDisplay() override {


        const float height = getHeight();
        const float width = getWidth();


        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(width , height ));

        if (ImGui::Begin("neural", nullptr, ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoTitleBar))
        {
            if(ImGui::Button("Randomise"))
            {
                getPluginDPSPointer()->randomise();
            }
            if(ImGui::Button("Delay##delaybutton"))
            {
                getPluginDPSPointer()->delay();
            }
            // if(ImGui::Button("Print matrix")){
            //     getPluginDPSPointer()->printMatrix();
            // }
            // if(ImGui::Button("Normalise")){
            //     getPluginDPSPointer()->normalise();
            // }
            float tempMaxEigenvalue=getPluginDPSPointer()->max_eigenvalue;
            if(ImGui::SliderFloat("Max eigenvalue", &tempMaxEigenvalue, 0.9f,1.f))
            {

                getPluginDPSPointer()->max_eigenvalue=tempMaxEigenvalue;
                needsNormalise=getPluginDPSPointer()->normalise();
            }

            if(needsNormalise) needsNormalise=getPluginDPSPointer()->normalise();


            ActivationFunctionType type=getPluginDPSPointer()->activation.load(std::memory_order_relaxed);
            // 2. Render the dropdown
            if (ImGui::BeginCombo("Activation function##UniqueLabel", activationFunctionNames[type]))
            {
                for (int i = 0; i < activationFunctionCount; ++i)
                {
                    const bool is_selected = (type == i);

                    // Render each item as selectable
                    if (ImGui::Selectable(activationFunctionNames[i], is_selected))
                    {
                        type = (ActivationFunctionType)i; // Update selection state on click
                    }

                    // Set the initial keyboard/scroll focus to the currently active selection
                    if (is_selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo(); // Always call EndCombo if BeginCombo returns true
            }
            getPluginDPSPointer()->activation.store(type,std::memory_order_relaxed);

            if(ImGui::SliderFloat("Delay##Delayslider",&fDelay,0.f,(float)MAX_DELAY))
            {
                if(ImGui::IsItemActivated())
                    editParameter(kParamDelay,true);
                setParameterValue(kParamDelay,(float)std::max(0,std::min(MAX_DELAY,(int)fDelay)));
            }

            if (ImGui::IsItemDeactivated())
            {
                editParameter(kParamDelay, false);
            }
        }
        //if(!ImGui::IsMouseDown(ImGuiMouseButton_Left)) endDrag();

        ImGui::End();
    }

    ~ImGuiPluginUI(){

    }

    DISTRHO_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ImGuiPluginUI)
};

// This must stay outside the class because it is a global framework entry point
UI* createUI()
{
    return new ImGuiPluginUI();
}

END_NAMESPACE_DISTRHO
