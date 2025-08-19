// E5Model.cs 최종 수정본

using UnityEngine;
using Microsoft.ML.Tokenizers;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Unity.InferenceEngine;

public class E5Model : MonoBehaviour
{
    public static E5Model Instance { get; private set; }

    [Header("E5 모델 파일")]
    public ModelAsset modelAsset;

    [Header("토크나이저 설정")]
    // [수정] 파일 이름 대신 TextAsset으로 직접 연결받도록 변경
    public TextAsset tokenizerAsset;

    private SentencePieceTokenizer tokenizer;
    private Worker engine;
    private bool isReady = false;

    void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
        DontDestroyOnLoad(gameObject);
        Initialize(); // Awake에서 바로 초기화를 호출
    }

    void Initialize()
    {
        // ONNX 모델 로딩
        var model = ModelLoader.Load(modelAsset);
        engine = new Worker(model, BackendType.GPUCompute);

        // [수정] 인스펙터에 연결된 tokenizerAsset이 있는지 확인
        if (tokenizerAsset != null)
        {
            // [수정] TextAsset의 byte 데이터를 메모리 스트림으로 변환하여 사용 (모든 플랫폼에서 작동)
            using (var modelStream = new MemoryStream(tokenizerAsset.bytes))
            {
                tokenizer = SentencePieceTokenizer.Create(modelStream);
            }
            isReady = true;
            Debug.Log("<color=cyan>E5 모델 및 토크나이저 준비 완료 (TextAsset 방식)</color>");
        }
        else
        {
            Debug.LogError("E5Model 인스펙터의 'Tokenizer Asset' 슬롯에 토크나이저(.bytes) 파일이 연결되지 않았습니다!");
            isReady = false;
        }
    }

    public float[] GetEmbedding(string text)
    {
        if (!isReady) { Debug.LogError("E5 모델이 준비되지 않았습니다. 초기화 에러를 확인하세요."); return null; }

        string prefixedText = "query: " + text;

        var rawIds = tokenizer.EncodeToIds(prefixedText, false, false);
        var processedIds = rawIds.Select(id => id + 1).ToList();
        processedIds.Insert(0, 0);
        processedIds.Add(2);

        var attentionMask = Enumerable.Repeat(1, rawIds.Count() + 2).ToList();
        var tokenTypeIds = Enumerable.Repeat(0, processedIds.Count).ToList();

        using var idsTensor = new Tensor<int>(new TensorShape(1, processedIds.Count), processedIds.ToArray());
        using var maskTensor = new Tensor<int>(new TensorShape(1, attentionMask.Count), attentionMask.ToArray());
        using var typeIdsTensor = new Tensor<int>(new TensorShape(1, tokenTypeIds.Count), tokenTypeIds.ToArray());

        engine.SetInput("input_ids", idsTensor);
        engine.SetInput("attention_mask", maskTensor);
        engine.SetInput("token_type_ids", typeIdsTensor);

        engine.Schedule();

        using var outputTensor = (engine.PeekOutput() as Tensor<float>).ReadbackAndClone();
        float[] rawOutput = outputTensor.DownloadToArray();

        return MeanPooling(rawOutput, attentionMask.ToArray(), outputTensor.shape);
    }

    private float[] MeanPooling(float[] rawOutput, int[] attentionMask, TensorShape shape)
    {
        int sequenceLength = shape[1];
        int features = shape[2];
        float[] meanPooled = new float[features];
        float sumMask = 0;

        for (int i = 0; i < sequenceLength; i++)
        {
            if (attentionMask[i] == 1)
            {
                sumMask += 1;
                for (int j = 0; j < features; j++)
                {
                    meanPooled[j] += rawOutput[i * features + j];
                }
            }
        }

        for (int i = 0; i < features; i++)
        {
            meanPooled[i] /= (sumMask + 1e-9f);
        }

        float norm = 0;
        for (int i = 0; i < features; i++)
        {
            norm += meanPooled[i] * meanPooled[i];
        }
        norm = Mathf.Sqrt(norm);

        for (int i = 0; i < features; i++)
        {
            meanPooled[i] /= (norm + 1e-9f);
        }

        return meanPooled;
    }

    void OnDestroy()
    {
        engine?.Dispose();
    }
}