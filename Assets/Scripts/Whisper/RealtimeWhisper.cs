using System.Collections.Generic;
using UnityEngine;
using Unity.InferenceEngine;
using System.Text;
using Unity.Collections;
using Newtonsoft.Json;
using System.Threading.Tasks;
using System;

public class RealtimeWhisper : MonoBehaviour
{
    // --- 모델 및 추론 관련 변수 ---
    Worker decoder1, decoder2, encoder, spectrogram;
    Worker argmax;

    // --- 토큰 관련 상수 및 변수 ---
    const int maxTokens = 100;
    const int END_OF_TEXT = 50257;
    const int START_OF_TRANSCRIPT = 50258;
    const int ENGLISH = 50259;
    const int KOREAN = 50264;
    const int GERMAN = 50261;
    const int FRENCH = 50265;
    const int TRANSCRIBE = 50359;
    const int TRANSLATE = 50358;
    const int NO_TIME_STAMPS = 50363;

    string[] tokens;
    int tokenCount = 0;
    NativeArray<int> outputTokens;
    int[] whiteSpaceCharacters = new int[256];
    public string outputString = "";

    // --- 텐서 관련 변수 ---
    Tensor<float> encodedAudio;
    bool transcribe = false;
    NativeArray<int> lastToken;
    Tensor<int> lastTokenTensor;
    Tensor<int> tokensTensor;
    Tensor<float> audioInput;

    // --- 에셋 파일 ---
    public ModelAsset audioDecoder1, audioDecoder2;
    public ModelAsset audioEncoder;
    public ModelAsset logMelSpectro;
    public TextAsset vocabAsset;

    [Header("명령어 처리기 연결")]
    public VoiceCommandProcessor commandProcessor;

    // --- 수정: Start() 메서드를 Awake()로 변경하고 모델 로딩만 담당 ---
    void Awake()
    {
        SetupWhiteSpaceShifts();
        GetTokens();

        decoder1 = new Worker(ModelLoader.Load(audioDecoder1), BackendType.GPUCompute);
        decoder2 = new Worker(ModelLoader.Load(audioDecoder2), BackendType.GPUCompute);

        FunctionalGraph graph = new FunctionalGraph();
        var input = graph.AddInput(DataType.Float, new DynamicTensorShape(1, 1, 51865));
        var amax = Functional.ArgMax(input, -1, false);
        var selectTokenModel = graph.Compile(amax);
        argmax = new Worker(selectTokenModel, BackendType.GPUCompute);

        encoder = new Worker(ModelLoader.Load(audioEncoder), BackendType.GPUCompute);
        spectrogram = new Worker(ModelLoader.Load(logMelSpectro), BackendType.GPUCompute);

        outputTokens = new NativeArray<int>(maxTokens, Allocator.Persistent);
        lastToken = new NativeArray<int>(1, Allocator.Persistent);
    }

    // --- 삭제: Update() 메서드 ---
    // 이 스크립트는 더 이상 스스로 업데이트할 필요가 없으므로 삭제합니다.

    // --- 핵심 기능: 외부에서 오디오 데이터를 받아 변환을 시작하는 공개 함수 ---
    public async Task StartTranscriptionFromAudioData(float[] audioData)
    {
        outputString = "";
        tokenCount = 3;
        outputTokens[0] = START_OF_TRANSCRIPT;
        outputTokens[1] = KOREAN;
        outputTokens[2] = TRANSCRIBE;

        Debug.Log($"[RealtimeWhisper] 오디오 데이터 수신 완료. 샘플 수: {audioData.Length}. 변환을 시작합니다.");

        // 오디오 데이터를 480000 크기로 패딩 (zero-padding)
        float[] paddedAudioData = new float[480000];
        Array.Copy(audioData, paddedAudioData, audioData.Length);

        // 수정된 audioInput 생성
        audioInput = new Tensor<float>(new TensorShape(1, 480000), paddedAudioData);


        //audioInput = new Tensor<float>(new TensorShape(1, audioData.Length), audioData);

        EncodeAudio();

        transcribe = true;
        tokensTensor = new Tensor<int>(new TensorShape(1, maxTokens));
        ComputeTensorData.Pin(tokensTensor);
        tokensTensor.Reshape(new TensorShape(1, tokenCount));
        tokensTensor.dataOnBackend.Upload<int>(outputTokens, tokenCount);

        lastToken[0] = NO_TIME_STAMPS;
        lastTokenTensor = new Tensor<int>(new TensorShape(1, 1), new[] { NO_TIME_STAMPS });

        while (transcribe && tokenCount < (outputTokens.Length - 1))
        {
            await InferenceStep();
        }

        if (!string.IsNullOrEmpty(outputString))
        {
            Debug.Log($"[RealtimeWhisper] 변환 완료: {outputString}. 명령어 처리기로 전달합니다.");
            commandProcessor?.ProcessTextCommand(outputString); // ?.는 null이 아닐 때만 실행
        }
        else
        {
            Debug.Log("[RealtimeWhisper] 변환 결과가 비어있어 처리하지 않습니다.");
        }
        transcribe = false;
    }

    void EncodeAudio()
    {
        spectrogram.Schedule(audioInput);
        var logmel = spectrogram.PeekOutput() as Tensor<float>;
        encoder.Schedule(logmel);
        encodedAudio = encoder.PeekOutput() as Tensor<float>;
    }

    async Task InferenceStep()
    {
        decoder1.SetInput("input_ids", tokensTensor);
        decoder1.SetInput("encoder_hidden_states", encodedAudio);
        decoder1.Schedule();

        var past_key_values_0_decoder_key = decoder1.PeekOutput("present.0.decoder.key") as Tensor<float>;
        var past_key_values_0_decoder_value = decoder1.PeekOutput("present.0.decoder.value") as Tensor<float>;
        var past_key_values_1_decoder_key = decoder1.PeekOutput("present.1.decoder.key") as Tensor<float>;
        var past_key_values_1_decoder_value = decoder1.PeekOutput("present.1.decoder.value") as Tensor<float>;
        var past_key_values_2_decoder_key = decoder1.PeekOutput("present.2.decoder.key") as Tensor<float>;
        var past_key_values_2_decoder_value = decoder1.PeekOutput("present.2.decoder.value") as Tensor<float>;
        var past_key_values_3_decoder_key = decoder1.PeekOutput("present.3.decoder.key") as Tensor<float>;
        var past_key_values_3_decoder_value = decoder1.PeekOutput("present.3.decoder.value") as Tensor<float>;

        var past_key_values_0_encoder_key = decoder1.PeekOutput("present.0.encoder.key") as Tensor<float>;
        var past_key_values_0_encoder_value = decoder1.PeekOutput("present.0.encoder.value") as Tensor<float>;
        var past_key_values_1_encoder_key = decoder1.PeekOutput("present.1.encoder.key") as Tensor<float>;
        var past_key_values_1_encoder_value = decoder1.PeekOutput("present.1.encoder.value") as Tensor<float>;
        var past_key_values_2_encoder_key = decoder1.PeekOutput("present.2.encoder.key") as Tensor<float>;
        var past_key_values_2_encoder_value = decoder1.PeekOutput("present.2.encoder.value") as Tensor<float>;
        var past_key_values_3_encoder_key = decoder1.PeekOutput("present.3.encoder.key") as Tensor<float>;
        var past_key_values_3_encoder_value = decoder1.PeekOutput("present.3.encoder.value") as Tensor<float>;

        decoder2.SetInput("input_ids", lastTokenTensor);
        decoder2.SetInput("past_key_values.0.decoder.key", past_key_values_0_decoder_key);
        decoder2.SetInput("past_key_values.0.decoder.value", past_key_values_0_decoder_value);
        decoder2.SetInput("past_key_values.1.decoder.key", past_key_values_1_decoder_key);
        decoder2.SetInput("past_key_values.1.decoder.value", past_key_values_1_decoder_value);
        decoder2.SetInput("past_key_values.2.decoder.key", past_key_values_2_decoder_key);
        decoder2.SetInput("past_key_values.2.decoder.value", past_key_values_2_decoder_value);
        decoder2.SetInput("past_key_values.3.decoder.key", past_key_values_3_decoder_key);
        decoder2.SetInput("past_key_values.3.decoder.value", past_key_values_3_decoder_value);

        decoder2.SetInput("past_key_values.0.encoder.key", past_key_values_0_encoder_key);
        decoder2.SetInput("past_key_values.0.encoder.value", past_key_values_0_encoder_value);
        decoder2.SetInput("past_key_values.1.encoder.key", past_key_values_1_encoder_key);
        decoder2.SetInput("past_key_values.1.encoder.value", past_key_values_1_encoder_value);
        decoder2.SetInput("past_key_values.2.encoder.key", past_key_values_2_encoder_key);
        decoder2.SetInput("past_key_values.2.encoder.value", past_key_values_2_encoder_value);
        decoder2.SetInput("past_key_values.3.encoder.key", past_key_values_3_encoder_key);
        decoder2.SetInput("past_key_values.3.encoder.value", past_key_values_3_encoder_value);

        decoder2.Schedule();

        var logits = decoder2.PeekOutput("logits") as Tensor<float>;
        argmax.Schedule(logits);
        using var t_Token = await argmax.PeekOutput().ReadbackAndCloneAsync() as Tensor<int>;
        int index = t_Token[0];

        outputTokens[tokenCount] = lastToken[0];
        lastToken[0] = index;
        tokenCount++;
        tokensTensor.Reshape(new TensorShape(1, tokenCount));
        tokensTensor.dataOnBackend.Upload<int>(outputTokens, tokenCount);
        lastTokenTensor.dataOnBackend.Upload<int>(lastToken, 1);

        if (index == END_OF_TEXT)
        {
            transcribe = false;
        }
        else if (index < tokens.Length)
        {
            outputString += GetUnicodeText(tokens[index]);
        }
    }

    void GetTokens()
    {
        var vocab = JsonConvert.DeserializeObject<Dictionary<string, int>>(vocabAsset.text);
        tokens = new string[vocab.Count];
        foreach (var item in vocab)
        {
            tokens[item.Value] = item.Key;
        }
    }

    string GetUnicodeText(string text)
    {
        var bytes = Encoding.GetEncoding("ISO-8859-1").GetBytes(ShiftCharacterDown(text));
        return Encoding.UTF8.GetString(bytes);
    }

    string ShiftCharacterDown(string text)
    {
        string outText = "";
        foreach (char letter in text)
        {
            outText += ((int)letter <= 256) ? letter : (char)whiteSpaceCharacters[(int)(letter - 256)];
        }
        return outText;
    }

    void SetupWhiteSpaceShifts()
    {
        for (int i = 0, n = 0; i < 256; i++)
        {
            if (IsWhiteSpace((char)i)) whiteSpaceCharacters[n++] = i;
        }
    }

    bool IsWhiteSpace(char c)
    {
        return !(('!' <= c && c <= '~') || ('¡' <= c && c <= '?') || ('®' <= c && c <= '?'));
    }

    private void OnDestroy()
    {
        decoder1?.Dispose();
        decoder2?.Dispose();
        encoder?.Dispose();
        spectrogram?.Dispose();
        argmax?.Dispose();
        audioInput?.Dispose();
        lastTokenTensor?.Dispose();
        tokensTensor?.Dispose();

        if (outputTokens.IsCreated) outputTokens.Dispose();
        if (lastToken.IsCreated) lastToken.Dispose();
    }
}