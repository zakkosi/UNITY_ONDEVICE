// VoiceCommandProcessor.cs
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;
using UnityEngine;

public class VoiceCommandProcessor : MonoBehaviour
{
    // [설정] 테스트할 명령어 에셋들을 Inspector 창에서 연결해주는 부분입니다.
    public VoiceCommandSO zoomIn;
    public VoiceCommandSO zoomOut;
    public VoiceCommandSO startRotation;
    public VoiceCommandSO stopRotation;
    public VoiceCommandSO showPopup;
    public VoiceCommandSO HidePopupCommand;
    public VoiceCommandSO reset;

    [Header("AI 설정")]
    [Range(0, 1)]
    [Tooltip("이 값 이상의 유사도를 가져야 명령으로 인정합니다.")]
    public float similarityThreshold = 0.8f;
    public VoiceUIController uiController;

    // [추가] 모든 명령어 에셋을 관리하기 편하게 리스트로 묶습니다.
    private List<VoiceCommandSO> allCommands;

    // [추가] 미리 계산된 키워드 임베딩을 저장할 딕셔너리
    // Key: 명령어 SO, Value: 해당 명령어의 키워드 임베딩 리스트
    private Dictionary<VoiceCommandSO, List<float[]>> commandEmbeddings;

    void Start()
    {
        // [추가] 앱 시작 시, 모든 명령어를 분석하여 임베딩을 미리 계산해 둡니다.
        InitializeCommands();
    }

    /*
    void Update()
    {
        // 숫자 1 키를 누르면 확대 명령(zoomIn)의 Invoke() 함수를 호출합니다.
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            Debug.Log("키보드 [1] 입력 -> '확대' 명령 테스트");
            zoomIn?.Invoke(); // null이 아닐 때만 실행
        }

        // 숫자 2 키를 누르면 축소 명령을 테스트합니다.
        if (Input.GetKeyDown(KeyCode.Alpha2))
        {
            Debug.Log("키보드 [2] 입력 -> '축소' 명령 테스트");
            zoomOut?.Invoke();
        }

        // 숫자 3 키를 누르면 회전 시작 명령을 테스트합니다.
        if (Input.GetKeyDown(KeyCode.Alpha3))
        {
            Debug.Log("키보드 [3] 입력 -> '회전 시작' 명령 테스트");
            startRotation?.Invoke();
        }

        // 숫자 4 키를 누르면 회전 정지 명령을 테스트합니다.
        if (Input.GetKeyDown(KeyCode.Alpha4))
        {
            Debug.Log("키보드 [4] 입력 -> '회전 정지' 명령 테스트");
            stopRotation?.Invoke();
        }

        // 숫자 5 키를 누르면 팝업 열기 명령을 테스트합니다.
        if (Input.GetKeyDown(KeyCode.Alpha5))
        {
            Debug.Log("키보드 [5] 입력 -> '팝업 열기' 명령 테스트");
            showPopup?.Invoke();
        }

        // 숫자 6 키를 누르면 팝업 열기 명령을 테스트합니다.
        if (Input.GetKeyDown(KeyCode.Alpha6))
        {
            Debug.Log("키보드 [6] 입력 -> '팝업 열기' 명령 테스트");
            HidePopupCommand?.Invoke();
        }

        // 숫자 0 키를 누르면 리셋 명령을 테스트합니다.
        if (Input.GetKeyDown(KeyCode.Alpha0))
        {
            Debug.Log("키보드 [0] 입력 -> '원래대로' 명령 테스트");
            reset?.Invoke();
        }
    }
    */
    public void ProcessTextCommand(string recognizedText)
    {
        Debug.Log("음성 인식 결과 수신: " + recognizedText);
        uiController?.AddToHistory(recognizedText);

        // 1. E5 모델로 현재 음성 텍스트의 임베딩을 가져옵니다.
        float[] recognizedTextEmbedding = E5Model.Instance.GetEmbedding(recognizedText);
        if (recognizedTextEmbedding == null) return;

        // 2. 가장 유사한 명령어를 찾기 위한 변수 초기화
        VoiceCommandSO bestMatchCommand = null;
        float highestSimilarity = 0f;

        // 3. 미리 계산해 둔 모든 명령어의 임베딩과 하나씩 비교
        foreach (var commandEntry in commandEmbeddings)
        {
            VoiceCommandSO command = commandEntry.Key;
            List<float[]> keywordEmbeddings = commandEntry.Value;

            foreach (var keywordEmbedding in keywordEmbeddings)
            {
                // 코사인 유사도(내적) 계산
                float similarity = CalculateSimilarity(recognizedTextEmbedding, keywordEmbedding);

                // 현재까지의 최고 점수보다 높으면, 이 명령을 최고 후보로 저장
                if (similarity > highestSimilarity)
                {
                    highestSimilarity = similarity;
                    bestMatchCommand = command;
                }
            }
        }

        // 4. 찾은 최고 점수가 설정한 임계값보다 높은지 확인
        if (highestSimilarity >= similarityThreshold)
        {
            Debug.Log($"<color=green>명령 실행: [{bestMatchCommand.name}] (유사도: {highestSimilarity:F2})</color>");
            bestMatchCommand.Invoke(); // 가장 비슷한 명령어 실행!
        }
        else
        {
            Debug.Log($"<color=yellow>적합한 명령을 찾지 못했습니다. (최고 유사도: {highestSimilarity:F2})</color>");
        }
    }

    /// <summary>
    /// 앱 시작 시 한 번만 호출되어 모든 키워드를 미리 벡터로 변환해 둡니다.
    /// </summary>
    private void InitializeCommands()
    {
        allCommands = new List<VoiceCommandSO>
        {
            zoomIn, zoomOut, startRotation, stopRotation, showPopup, HidePopupCommand, reset
        };

        commandEmbeddings = new Dictionary<VoiceCommandSO, List<float[]>>();
        Debug.Log("명령어 임베딩 초기화를 시작합니다...");

        foreach (var command in allCommands)
        {
            if (command == null) continue;

            var keywordEmbeddings = new List<float[]>();
            foreach (var keyword in command.keywords)
            {
                // E5Model을 사용해 각 키워드의 임베딩을 계산
                var embedding = E5Model.Instance.GetEmbedding(keyword);
                if (embedding != null)
                {
                    keywordEmbeddings.Add(embedding);
                }
            }
            commandEmbeddings[command] = keywordEmbeddings;
        }
        Debug.Log("명령어 임베딩 완료!");
    }

    /// <summary>
    /// 두 벡터의 코사인 유사도(내적)를 계산합니다.
    /// </summary>
    private float CalculateSimilarity(float[] vecA, float[] vecB)
    {
        if (vecA == null || vecB == null || vecA.Length != vecB.Length)
        {
            return 0;
        }

        // E5 임베딩은 이미 정규화(Normalized)되어 있으므로, 내적이 곧 코사인 유사도입니다.
        return Enumerable.Range(0, vecA.Length).Sum(i => vecA[i] * vecB[i]);
    }
}