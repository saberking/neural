#ifndef PLUGINDSP_HPP
#define PLUGINDSP_HPP

#include "DistrhoPlugin.hpp"
#include "Parameters.hpp"
#include "WinConsoleOutput.hpp"
#include "external/base64.h"
#include "Undo.hpp"
#include "external/Eigen/Dense"
#include <atomic>
#include <ctime>
#include "external/Eigen/SVD" // Make sure to include the SVD header at the top of your file
#include <random> // <-- ADD THIS LINE HERE
#include <chrono>

START_NAMESPACE_DISTRHO
    enum ActivationFunctionType{
    activationFunctionClip,
    activationFunctionTanh,
    activationFunctionNone,
    activationFunctionCount
};
static const char* activationFunctionNames[]={
        "Clip",
        "Tanh",
        "None"
    };

#define OUT_SIZE 30
#define CONSTANT_KNOB_COUNT 4
#define STRIDE (OUT_SIZE+CONSTANT_KNOB_COUNT)
#define MAX_DELAY 1000
class ImGuiPluginDSP : public Plugin
{
    float fA = 0.0f;
    float fB=0.f;
    float fC=0.f;
    float fD=0.f;
    float fDelay=0.f;
    bool consoleAttached=false;
    std::mt19937 gen;
    std::normal_distribution<float> d;
    float note=1.f,lastNote=1.f;
public:
    std::atomic<ActivationFunctionType> activation=activationFunctionClip;
    float max_eigenvalue=1.f;
    std::atomic<bool> updateReady=false;
    int lastDelay=0;

    UndoItem *undoItems[MAX_UNDO_DEPTH];
    int nextUndoIndex=0;
    int undoCount=0,redoCount=0;
    alignas(64) float weightBuffer1[OUT_SIZE*(OUT_SIZE+CONSTANT_KNOB_COUNT)];
    alignas(64) float weightBuffer2[OUT_SIZE*(OUT_SIZE+CONSTANT_KNOB_COUNT)];
    std::atomic<float *> weightBufferPointer;
    alignas(64) float inputBuffer[MAX_DELAY+1][OUT_SIZE+CONSTANT_KNOB_COUNT];
    int inputBufferIndex=0;
    alignas(64) float outputBuffer[OUT_SIZE];
    ImGuiPluginDSP()
        : Plugin(kParamCount, 0, 1) // parameters, programs, states
    {
        if (DEBUG&&!GetConsoleWindow()) {
            initConsoleOutput();
            consoleAttached=true;
        }

        for(int i=0;i<MAX_UNDO_DEPTH;i++)
        {
            undoItems[i]=NULL;
        }

        for(int i=0;i<OUT_SIZE*(OUT_SIZE+CONSTANT_KNOB_COUNT);i++)
        {
            weightBuffer1[i]=0.f;
        }

        for(int i=0;i<MAX_DELAY+1;i++)
        {
            for(int j=0;j<OUT_SIZE+CONSTANT_KNOB_COUNT;j++)
            {
               inputBuffer[i][j]=0.f;

            }
        }
        std::srand(std::time(nullptr));
        std::random_device rd;
        gen.seed(rd()); // Seed this specific instance with a hardware random

        Eigen::Map<Eigen::Matrix<float, OUT_SIZE, OUT_SIZE+CONSTANT_KNOB_COUNT, Eigen::RowMajor>> W(weightBuffer1);
        W.diagonal().setConstant(1.f);
        weightBufferPointer.store(weightBuffer1,std::memory_order_release);
    }

    float findMaxAmplification(float* outPointer)
    {
        Eigen::Map<Eigen::Matrix<float, OUT_SIZE, OUT_SIZE+CONSTANT_KNOB_COUNT, Eigen::RowMajor>> W(outPointer);

        // Isolate the square audio feedback block (e.g., 24x24)
        Eigen::Matrix<float, OUT_SIZE, OUT_SIZE> audioBlock = W.block<OUT_SIZE, OUT_SIZE>(0, 0);

        // Compute the Singular Value Decomposition (SVD)
        // We only need the singular values, so we pass 0 to skip computing U and V matrices (saves CPU)
        Eigen::JacobiSVD<Eigen::Matrix<float, OUT_SIZE, OUT_SIZE>> svd(audioBlock, 0);

        // The singular values are always returned sorted from highest to lowest.
        // Index 0 is mathematically guaranteed to be the maximum amplification factor!
        float maxAmplification = svd.singularValues()[0];

        std::cout << "Absolute Highest Single-Hop Amplification Factor: " << maxAmplification << std::endl;

        return maxAmplification;
    }


    void normaliseMatrix(float *outPointer)
    {
        Eigen::Map<Eigen::Matrix<float, OUT_SIZE, OUT_SIZE+CONSTANT_KNOB_COUNT, Eigen::RowMajor>> W(outPointer);

        Eigen::EigenSolver<Eigen::Matrix<float, OUT_SIZE-2, OUT_SIZE-2>> solver(W.block<OUT_SIZE-2, OUT_SIZE-2>(2, 2), false);
        float maxMagnitude = 0.0f;
        for (int i = 0; i < OUT_SIZE-2; ++i) {
            maxMagnitude = std::max(maxMagnitude, std::abs(solver.eigenvalues()[i]));
        }
        std::cout<<"normalise"<<maxMagnitude<<std::endl;
        //maxMagnitude+=0.001f;
        float mult=max_eigenvalue/std::max(0.001f,maxMagnitude);

        //float mult=1/findMaxAmplification(outPointer);

        W *=mult;

        // for (int r = 0; r < OUT_SIZE; ++r) {
        //     for (int c = 0; c < OUT_SIZE; ++c) {
        //         if (std::abs(W(r, c)) > MAX_EIGENVALUE) {
        //             // Keep the original positive/negative sign but snap the value to 0.95f
        //             W(r, c) = std::copysign(MAX_EIGENVALUE, W(r, c));
        //         }
        //     }
        // }
        W.block<OUT_SIZE,CONSTANT_KNOB_COUNT>(0,OUT_SIZE).setZero();
        W.block<2,OUT_SIZE>(0,0).rowwise().normalize();
        //         W.block<OUT_SIZE,2>(0,0)*=0.95f;
        // W.block<2,OUT_SIZE>(0,0)*=0.95f;


        printEigen(outPointer);
    }

    void printEigen(float *outPointer)
    {
        Eigen::Map<Eigen::Matrix<float, OUT_SIZE, OUT_SIZE+CONSTANT_KNOB_COUNT, Eigen::RowMajor>> W(outPointer);

        Eigen::EigenSolver<Eigen::Matrix<float, OUT_SIZE-2, OUT_SIZE-2>> solver(W.block<OUT_SIZE-2, OUT_SIZE-2>(2, 2), false);
        float maxMagnitude = 0.0f;
        for (int i = 0; i < OUT_SIZE-2; ++i) {
            maxMagnitude = std::max(maxMagnitude, std::abs(solver.eigenvalues()[i]));
        }
        std::cout<<"maxMAgnitude "<<maxMagnitude<<std::endl;
        //float mult=1/findMaxAmplification(outPointer);

    }

    bool normalise()
    {
        if(updateReady.load(std::memory_order_acquire))return true;
        float *inPointer, *outPointer;
        inPointer=weightBufferPointer.load(std::memory_order_acquire);
        outPointer=(inPointer==weightBuffer1?weightBuffer2:weightBuffer1);

        for(int i=0;i<OUT_SIZE*(OUT_SIZE+CONSTANT_KNOB_COUNT);i++)
        {
            outPointer[i]=inPointer[i];//+(i%3?0.1:-0.1);
        }
        normaliseMatrix(outPointer);

        updateReady.store(true,std::memory_order_release);
        return false;

    }
    void randomise()
    {
        if(updateReady.load(std::memory_order_acquire))return;
        float *inPointer, *outPointer;
        inPointer=weightBufferPointer.load(std::memory_order_acquire);
        outPointer=(inPointer==weightBuffer1?weightBuffer2:weightBuffer1);

        for(int i=0;i<OUT_SIZE*(OUT_SIZE+CONSTANT_KNOB_COUNT);i++)
        {
            outPointer[i]=inPointer[i];//+(i%3?0.1:-0.1);
        }



        Eigen::Map<Eigen::Matrix<float, OUT_SIZE, OUT_SIZE+CONSTANT_KNOB_COUNT, Eigen::RowMajor>> W(outPointer);

        Eigen::MatrixXf X(OUT_SIZE, OUT_SIZE);
        for (int r = 0; r < OUT_SIZE; ++r) {
            for (int c = 0; c < OUT_SIZE; ++c) {
                X(r, c) = d(gen); // Populating matrix with a true Gaussian profile
            }
        }

        // A Householder QR acting on a Gaussian matrix produces a perfectly un-biased
        // Haar-distributed random orthogonal matrix, eliminating Left/Right panning bias!
        Eigen::HouseholderQR<Eigen::MatrixXf> qr(X);
        Eigen::MatrixXf Q = qr.householderQ();

        W.block<OUT_SIZE, OUT_SIZE>(0, 0) = (W.block<OUT_SIZE, OUT_SIZE>(0, 0) * Q * 0.1f).eval()
                                            + (W.block<OUT_SIZE, OUT_SIZE>(0, 0) * 0.9f).eval();

        // Fix noise injection using the same Gaussian distribution profile
        Eigen::MatrixXf noise(OUT_SIZE, OUT_SIZE);
        const float noiseAmount = 0.05f;
        for (int r = 0; r < OUT_SIZE; ++r) {
            for (int c = 0; c < OUT_SIZE; ++c) {
                noise(r, c) = d(gen) * noiseAmount;
            }
        }
    W.block<OUT_SIZE, OUT_SIZE>(0, 0) += noise;
        normaliseMatrix(outPointer);

        updateReady.store(true,std::memory_order_release);

    }
    void delay()
    {
        if(updateReady.load(std::memory_order_acquire)) return;
        float *inPointer, *outPointer;
        inPointer=weightBufferPointer.load(std::memory_order_acquire);
        outPointer=(inPointer==weightBuffer1?weightBuffer2:weightBuffer1);

        for(int i=0;i<OUT_SIZE*(OUT_SIZE+CONSTANT_KNOB_COUNT);i++)
        {
            outPointer[i]=inPointer[i];//+(i%3?0.1:-0.1);
        }


        // outPointer[OUT_SIZE-2]=1.f;
        // outPointer[(OUT_SIZE+CONSTANT_KNOB_COUNT)+OUT_SIZE-1]=1.f;
        // for(int i =2;i<OUT_SIZE;i++)
        // {
        //     outPointer[(OUT_SIZE+CONSTANT_KNOB_COUNT)*i+i-2]=1.f;

        // }
        Eigen::Map<Eigen::Matrix<float, OUT_SIZE, OUT_SIZE+CONSTANT_KNOB_COUNT, Eigen::RowMajor>> W(outPointer);


        auto circularSequence = (Eigen::VectorXi::LinSpaced(OUT_SIZE, -2, OUT_SIZE - 3).array() + OUT_SIZE)
                                    .unaryExpr([](int val) { return val % OUT_SIZE; });

        // 2. COMPILER-SAFE REWRITE: Create a temporary copy of the old matrix state
        // This completely stops memory aliasing bugs without needing Eigen::all!
        Eigen::Matrix<float, OUT_SIZE, OUT_SIZE + CONSTANT_KNOB_COUNT, Eigen::RowMajor> tempW = W;

        // 3. Remap the rows sequentially
        for (int i = 0; i < OUT_SIZE; ++i) {
            int targetRowIndex = circularSequence[i];
            W.row(i) += tempW.row(targetRowIndex)*0.2f; // Cleanly assigns the whole row
        }

        normaliseMatrix(outPointer);

        updateReady.store(true,std::memory_order_release);



    }



    void printMatrix()
    {
        for(int i=0;i<OUT_SIZE;i++){
            for(int j=0;j<OUT_SIZE+CONSTANT_KNOB_COUNT;j++)        std::cout<<weightBufferPointer.load(std::memory_order_acquire)[i*(OUT_SIZE+CONSTANT_KNOB_COUNT)+j]<<" ";
            std::cout<<std::endl;
        }
        printEigen(weightBufferPointer.load(std::memory_order_acquire));
    }
    ~ImGuiPluginDSP(){
        for(int i=0;i<MAX_UNDO_DEPTH;i++)
        {
            if(undoItems[i])delete undoItems[i];
        }
    }

protected:




    float clip(float a, float abs=1.f)
    {
        return std::max(-1.f*abs,std::min(1.f*abs,a));
    }
    inline float fastTanh(float x) {
        // Clamp input to prevent polynomial divergence at high values
        float x_clamped = std::max(-4.5f, std::min(4.5f, x));
        float x2 = x_clamped * x_clamped;

        // Pade approximation: highly accurate, no division loops, zero hardware stalls
        return x_clamped * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2))) /
               (135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f)));
    }
    void noteOn(int midiNote, float velocity){
        lastNote=note;
        note=std::pow(2.f, -(float)midiNote/12.f);
    }

    void noteOff(int midiNote){

    }

    void handleMidi(const MidiEvent *midiEvent){
        int status = midiEvent->data[0]; // midi status
        int midi_message = status & 0xF0;
        int midi_data1 = midiEvent->data[1];
        int midi_data2 = midiEvent->data[2];
        float velocity=(float)midi_data2;
        velocity/=128.f;
        switch ( midi_message )
        {
        case 0x80: // note_off
            noteOff(midi_data1);
            break;
        case 0x90: // note_on
            noteOn(midi_data1, velocity);
            break;
        }
    }
    std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
    float time1=0, time2=0, time3=0, time4=0, time5=0, time6=0, time7=0, time8=0;
    int framecounter=0;
    void printTime()
    {
        std::cout<<"Time"<<std::endl;
        std::cout<<time1<<"    "<<time2<<"   "<<time3<<"    "<<time4<<"   "<<time5<<"    "<<time6<<"   "<<time7<<"    "<<time8<<std::endl;
    }
    void startTimer(){
        start= std::chrono::steady_clock::now();
    }
    float getTimeInterval(){
        std::chrono::steady_clock::time_point end;

        end= std::chrono::steady_clock::now();
        // 1. Get seconds as a float (e.g., 0.01234 seconds)
        float seconds = std::chrono::duration<float>(end - start).count();
        float milliseconds = std::chrono::duration<float, std::milli>(end - start).count();
        startTimer();
        return milliseconds;
    }

    void run ( const float **inputs, float **outputs, uint32_t frames,
             const MidiEvent *midiEvents, // MIDI pointer
             uint32_t midiEventCount      // Number of MIDI events in block
             ) override
    {
        startTimer();
        int curEventIndex =0;

        ActivationFunctionType activationFunction=activation.load(std::memory_order_release);

        if(updateReady.load(std::memory_order_acquire))
        {
            float *newPointer=(weightBufferPointer.load(std::memory_order_acquire)==weightBuffer1?weightBuffer2:weightBuffer1);
            weightBufferPointer.store(newPointer,std::memory_order_release);
            updateReady.store(false,std::memory_order_release);

        }
        int delay=(int)std::max(0.f,std::min(fDelay*note,(float)MAX_DELAY));
        int currentLoopDelay=lastDelay;
        int lastLoopDelay=currentLoopDelay;

        time1+=getTimeInterval();

        Eigen::Map<Eigen::Matrix<float, OUT_SIZE, OUT_SIZE+CONSTANT_KNOB_COUNT, Eigen::RowMajor>, Eigen::Aligned64>
            W(weightBufferPointer.load(std::memory_order_acquire));

        // Eigen::Map<Eigen::Matrix<float, OUT_SIZE, OUT_SIZE+CONSTANT_KNOB_COUNT,
        //                          Eigen::RowMajor>> W(weightBufferPointer.load(std::memory_order_relaxed));

        Eigen::Map<Eigen::Vector<float, OUT_SIZE>> y(outputBuffer);
        time2+=getTimeInterval();
        for (uint32_t sample = 0; sample < frames; ++sample) {
            while ( curEventIndex < midiEventCount && sample == midiEvents[curEventIndex].frame )
            {
                handleMidi(&(midiEvents[curEventIndex++]));

            }
            if(note!=lastNote)
            {
                delay=(int)std::max(0.f,std::min(fDelay*note,(float)MAX_DELAY));
                lastNote=note;
            }
            int temp=delay-lastDelay;
            int temp2=(temp*((int)sample+1))/(int)frames;

            currentLoopDelay=lastDelay+temp2;
            int newIndex=(inputBufferIndex+lastLoopDelay-currentLoopDelay+MAX_DELAY+1)%(MAX_DELAY+1);
            if(newIndex<0)std::cout<<"ERRORR!!!!!"<<"ERRRORRRR!!!!!!"<<std::endl<<"current "<<
                          currentLoopDelay<<" lastD "<<lastDelay<<"   delay "<<delay<<"    sammple "<<sample<<"    frames"<<frames<<
                    "    temp"<<temp<<"    temp2 "<<temp2<<  std::endl;
            else inputBufferIndex=newIndex;
            lastLoopDelay=currentLoopDelay;

            time3+=getTimeInterval();
            Eigen::Map<Eigen::Vector<float, OUT_SIZE+CONSTANT_KNOB_COUNT>> x(inputBuffer[inputBufferIndex]);
            // Append your 4 special parameters to the remaining 4 slots of x
            x[OUT_SIZE] = fA;
            x[OUT_SIZE+1] = fB;
            x[OUT_SIZE+2] = fC;
            x[OUT_SIZE+3] = fD;
            // 1. Load your sample into your input vector 'x' here...
            x[0]=inputs[0][sample];x[1]=inputs[1][sample];

            time4+=getTimeInterval();
            const float* rawW = W.data();
            const float* rawX = x.data();
            float*       rawY = y.data();


            // 2. Your ultra-fast, thread-safe unrolled loop compiles perfectly now!
            time5+=getTimeInterval();
            for (int r = 0; r < OUT_SIZE; ++r) {
                float sum = 0.0f;

                // Use rawW instead of the Eigen object W
                const float* rowPtr = &rawW[r * STRIDE];

                for (int c = 0; c < STRIDE; ++c) {
                    sum += rowPtr[c] * rawX[c];
                }

                // if(r>2&&sum<-0.5f)
                // {
                //         sum=sum*0.8-0.1f;

                // }

                rawY[r] = sum;
            }
            time6+=getTimeInterval();
            if(activationFunction==activationFunctionClip)
            {
                y = y.array().cwiseMax(-1.0f).cwiseMin(1.0f);

            }
            if(activationFunction==activationFunctionTanh)
            {
                float* rawY = y.data();
                for (int r = 0; r < OUT_SIZE; ++r) {
                    rawY[r] = fastTanh(rawY[r]);
                }
            }
            time7+=getTimeInterval();


            outputs[0][sample]=clip(y[0],2);outputs[1][sample]=clip(y[1],2);

            inputBufferIndex=(inputBufferIndex+1)%(MAX_DELAY+1);
            Eigen::Map<Eigen::Vector<float, OUT_SIZE+CONSTANT_KNOB_COUNT>> x2(inputBuffer[(inputBufferIndex+currentLoopDelay)%(MAX_DELAY+1)]);
            x2.head<OUT_SIZE>() = y.eval();
            time8+=getTimeInterval();
            framecounter++;

        }
        lastDelay=lastLoopDelay;
        if(!(framecounter%48000)) printTime();
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Information
    /**
      Initialize the parameter @a index.@n
      This function will be called once, shortly after the plugin is created.
    */
    void initParameter(uint32_t index, Parameter& parameter) override
    {
        if(index==kParamA)
        {
            parameter.ranges.min = -1.f;
            parameter.ranges.max = 1.f;
            parameter.ranges.def = 0.f;
            parameter.name = "A";
            parameter.symbol = "A";
            parameter.hints=kParameterIsAutomatable;
        }
        if(index==kParamB)
        {
            parameter.ranges.min = -1.f;
            parameter.ranges.max = 1.f;
            parameter.ranges.def = 0.f;
            parameter.name = "B";
            parameter.symbol = "B";
            parameter.hints=kParameterIsAutomatable;
        }
        if(index==kParamC)
        {
            parameter.ranges.min = -1.f;
            parameter.ranges.max = 1.f;
            parameter.ranges.def = 0.f;
            parameter.name = "C";
            parameter.symbol = "C";
            parameter.hints=kParameterIsAutomatable;
        }
        if(index==kParamD)
        {
            parameter.ranges.min = -1.f;
            parameter.ranges.max = 1.f;
            parameter.ranges.def = 0.f;
            parameter.name = "D";
            parameter.symbol = "D";
            parameter.hints=kParameterIsAutomatable;
        }
        if(index==kParamDelay)
        {
            parameter.ranges.min = 0.f;
            parameter.ranges.max = MAX_DELAY;
            parameter.ranges.def = 0.f;
            parameter.name = "Delay";
            parameter.symbol = "Delay";
            parameter.hints=kParameterIsAutomatable;
        }


    }

    float getParameterValue(uint32_t index) const override
    {
        if(index==kParamA){
            return fA;
        }
        if(index==kParamB){
            return fB;
        }
        if(index==kParamC){
            return fC;
        }
        if(index==kParamD){
            return fD;
        }
        if(index==kParamDelay){
            return fDelay;
        }
    }


    void setParameterValue(uint32_t index, float value) override
    {
        if(index==kParamA){
            fA=value;
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
    }

    void activate() override
    {
    }

    void initState(uint32_t index, String& key, String& defaultValue) override
    {
        if (index == 0) {
            key = "sampleData";
            defaultValue = "";
        }
    }

    String getState(const char* key) const override {
        if (!strcmp(key,"sampleData"))
        {

            size_t maxEigenSize=sizeof(float);
            size_t activationSize=sizeof(ActivationFunctionType);
            size_t matrixSize=sizeof(float)*(OUT_SIZE+CONSTANT_KNOB_COUNT)*OUT_SIZE;

            size_t totalBytes = maxEigenSize+activationSize+matrixSize;

            std::vector<uint8_t> rawBinaryBuffer(totalBytes);
            auto *incrementalPointer=rawBinaryBuffer.data();

            auto *maxEigenPointer=reinterpret_cast<float*>(incrementalPointer);
            *maxEigenPointer=max_eigenvalue;
            incrementalPointer+=maxEigenSize;

            auto* activationPointer=reinterpret_cast<ActivationFunctionType*>(incrementalPointer);
            *activationPointer=activation.load(std::memory_order_relaxed);
            incrementalPointer+=activationSize;

            auto *matrixPtr=reinterpret_cast<float*>(incrementalPointer);
            auto *buffer=weightBufferPointer.load(std::memory_order_relaxed);
            for(int i=0;i<OUT_SIZE*(OUT_SIZE+CONSTANT_KNOB_COUNT);i++)
            {
                matrixPtr[i]=buffer[i];
            }

            std::string encodedText = base64_encode(rawBinaryBuffer.data(), rawBinaryBuffer.size());
            return String(encodedText.c_str());
        }else return String("");

    }

    void setState(const char *key, const char * value){
        if(!strcmp("sampleData", key))
        {
            if (strlen(value) == 0 ) {
                return;
            }


            std::string decodedBytes = base64_decode(std::string(value));

            const uint8_t* incrementalPointer = reinterpret_cast<const uint8_t*>(decodedBytes.data());

            size_t maxEigenSize=sizeof(float);
            size_t activationSize=sizeof(ActivationFunctionType);
            size_t matrixSize=sizeof(float)*(OUT_SIZE+CONSTANT_KNOB_COUNT)*OUT_SIZE;


            auto *maxEigenPointer=reinterpret_cast<const float*>(incrementalPointer);
            max_eigenvalue=*maxEigenPointer;
            incrementalPointer+=maxEigenSize;

            auto* activationPointer=reinterpret_cast<const ActivationFunctionType*>(incrementalPointer);
            activation.store(*activationPointer,std::memory_order_relaxed);
            incrementalPointer+=activationSize;

            auto *matrixPtr=reinterpret_cast<const float*>(incrementalPointer);
            updateReady.store(false,std::memory_order_release);
            auto *buffer=weightBufferPointer.load(std::memory_order_acquire);
            float *outPointer=(buffer==weightBuffer1?weightBuffer2:weightBuffer1);
            for(int i=0;i<OUT_SIZE*(OUT_SIZE+CONSTANT_KNOB_COUNT);i++)
            {
                outPointer[i]=matrixPtr[i];
            }
            weightBufferPointer.store(outPointer,std::memory_order_release);

        }
    }
    /**
      Get the plugin label.@n
      This label is a short restricted name consisting of only _, a-z, A-Z and 0-9 characters.
    */
    const char* getLabel() const noexcept override
    {
        return "neural";
    }

    /**
      Get an extensive comment/description about the plugin.@n
      Optional, returns nothing by default.
    */
    const char* getDescription() const override
    {
        return "neural FX";
    }

    /**
      Get the plugin author/maker.
    */
    const char* getMaker() const noexcept override
    {
        return "Jean Pierre Cimalando, falkTX, Saber";
    }

    /**
      Get the plugin license (a single line of text or a URL).@n
      For commercial plugins this should return some short copyright information.
    */
    const char* getLicense() const noexcept override
    {
        return "ISC";
    }

    /**
      Get the plugin version, in hexadecimal.
      @see d_version()
    */
    uint32_t getVersion() const noexcept override
    {
        return d_version(1, 0, 0);
    }

    /**
      Get the plugin unique Id.@n
      This value is used by LADSPA, DSSI and VST plugin formats.
      @see d_cconst()
    */
    int64_t getUniqueId() const noexcept override
    {
        return d_cconst('n', 'e', 'u', 'r');
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Init
    DISTRHO_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ImGuiPluginDSP)
};

END_NAMESPACE_DISTRHO

#endif // PLUGINUI_HPP
