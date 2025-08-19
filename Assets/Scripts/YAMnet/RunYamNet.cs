using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;

// --- 핵심 기능만 남긴 RunYamNet 스크립트 ---
public class RunYamNet : MonoBehaviour
{
    private const float AudioBufferLengthSec = 0.96f;
    public const int sampleRate = 16000; // YAMNet expects 16kHz audio

    private const int NumClasses = 521;
    public ModelAsset modelAsset;
    public TextAsset classMapData;

    private Unity.InferenceEngine.Model model;
    private Unity.InferenceEngine.Model modelWithArgMax;
    private Unity.InferenceEngine.Worker worker;

    private string[] classMap;

    private const int SPEECH_ID = 0;
    private const int SILENCE_ID = 494; // 필요시 사용 가능

    private bool isFirstStarted = false;

    private float[] audioBuffer = new float[audioBufferSize];
    private const int audioBufferSize = (int)(AudioBufferLengthSec * sampleRate);

    // --- 음성 녹음 관련 변수 ---
    private bool isRecordingSpeech = false;
    private readonly List<float> speechAudioBuffer = new List<float>();
    private int previousLabelIndex = -1;
    private float[] previousAudioBuffer;

    // --- 수정: 온디바이스 Whisper 스크립트 참조만 남김 ---
    public RealtimeWhisper onDeviceWhisper;
    public VoiceUIController uiController;

    private bool isShuttingDown = false;

    void Awake()
    {
        previousAudioBuffer = new float[audioBufferSize];

        if (modelAsset)
        {
            model = Unity.InferenceEngine.ModelLoader.Load(modelAsset);
            var graph = new Unity.InferenceEngine.FunctionalGraph();
            var inputs = graph.AddInputs(model);
            Unity.InferenceEngine.FunctionalTensor[] outputs = Unity.InferenceEngine.Functional.Forward(model, inputs);
            Unity.InferenceEngine.FunctionalTensor argmaxOutput = Unity.InferenceEngine.Functional.ArgMax(outputs[0], -1, false);
            modelWithArgMax = graph.Compile(argmaxOutput);
            worker = new Unity.InferenceEngine.Worker(modelWithArgMax, Unity.InferenceEngine.BackendType.CPU);
        }
        else
        {
            Debug.LogWarning("[RunYamNet] ModelAsset is not assigned in the Inspector!");
        }

        classMap = new string[NumClasses];
        LoadClassMap();
    }

    private bool isDataUpdated = false;

    public void AppendAudio(float[] data, bool enableFlag)
    {
        Array.Copy(data, 0, audioBuffer, 0, audioBufferSize);
        isDataUpdated = true;
        isFirstStarted = enableFlag;
    }

    async void LateUpdate()
    {
        if (!isFirstStarted || !isDataUpdated || isProcessing) return;

        isProcessing = true;
        try
        {
            isDataUpdated = false;
            int currentLabelIndex = await Inference(audioBuffer);

            if (previousLabelIndex != SPEECH_ID && currentLabelIndex == SPEECH_ID)
            {
                uiController?.SetStatusListening();
                Debug.Log("[RunYamNet] >>> Speech detected. Starting audio recording.");
                isRecordingSpeech = true;
                speechAudioBuffer.Clear();
                speechAudioBuffer.AddRange(previousAudioBuffer);
            }

            if (isRecordingSpeech)
            {
                speechAudioBuffer.AddRange(audioBuffer);
            }

            // --- 수정: 파일명 관련 로직 삭제 ---
            if (previousLabelIndex == SPEECH_ID && currentLabelIndex != SPEECH_ID)
            {
                isRecordingSpeech = false;
                Debug.Log("[RunYamNet] <<< Speech ended. Passing data to on-device Whisper.");
                await StoreSpeechClip();
            }

            Array.Copy(audioBuffer, previousAudioBuffer, audioBufferSize);
            previousLabelIndex = currentLabelIndex;
        }
        catch (Exception e)
        {
            Debug.LogError($"An error occurred in LateUpdate: {e.ToString()}");
        }
        finally
        {
            isProcessing = false;
        }
    }

    private bool isProcessing = false;

    private async Task<int> Inference(float[] resampledBuffer)
    {
        if (isShuttingDown || worker == null) return -1;

        using var inputTensor = new Unity.InferenceEngine.Tensor<float>(new Unity.InferenceEngine.TensorShape(resampledBuffer.Length), resampledBuffer);
        worker.SetInput(modelWithArgMax.inputs[0].name, inputTensor);
        worker.Schedule();
        using var outputTensor = await worker.PeekOutput(modelWithArgMax.outputs[0].name).ReadbackAndCloneAsync() as Unity.InferenceEngine.Tensor<int>;
        return outputTensor[0];
    }

    // --- 수정: 온디바이스 Whisper 호출 로직만 남도록 대폭 수정 ---
    private async Task StoreSpeechClip()
    {
        uiController?.SetStatusProcessing();
        if (speechAudioBuffer.Count == 0) return;

        // 1. 온디바이스 Whisper 스크립트가 연결되었는지 확인
        if (onDeviceWhisper == null)
        {
            Debug.LogError("On-Device Whisper 스크립트가 연결되지 않았습니다!");
            return;
        }

        try
        {
            // 2. 녹음된 오디오 버퍼를 float[] 배열로 변환
            float[] recordedAudio = speechAudioBuffer.ToArray();
            Debug.Log($"[RunYamNet] 오디오 녹음 완료. 샘플 수: {recordedAudio.Length}. 온디바이스 Whisper로 전달합니다.");

            // 3. 온디바이스 Whisper의 함수를 직접 호출
            await onDeviceWhisper.StartTranscriptionFromAudioData(recordedAudio);

            speechAudioBuffer.Clear();
        }
        catch (Exception e)
        {
            Debug.LogError($"StoreSpeechClip failed: {e.ToString()}");
        }
    }

    private void LoadClassMap()
    {
        if (classMapData == null) return;
        using (var reader = new StringReader(classMapData.text))
        {
            string line = reader.ReadLine(); // 헤더 무시
            while ((line = reader.ReadLine()) != null)
            {
                if (!string.IsNullOrWhiteSpace(line))
                {
                    string[] parts = line.Split(',');
                    if (parts.Length >= 3 && int.TryParse(parts[0], out int classId) && classId < classMap.Length)
                    {
                        classMap[classId] = parts[2].Trim('"');
                    }
                }
            }
        }
    }

    private void OnDestroy()
    {
        isShuttingDown = true;
        worker?.Dispose();
    }
}