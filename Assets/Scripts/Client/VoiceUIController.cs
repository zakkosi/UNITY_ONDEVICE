// VoiceUIController.cs
using UnityEngine;
using TMPro; // TextMeshPro를 사용하기 위해 추가
using System.Collections.Generic;

public class VoiceUIController : MonoBehaviour
{
    [Header("UI 연결")]
    public GameObject historyContent; // 스크롤 뷰의 Content 오브젝트
    public GameObject historyEntryPrefab; // 히스토리 항목으로 사용할 Text 프리팹
    public TextMeshProUGUI statusText; // 현재 상태를 표시할 Text

    private List<string> commandHistory = new List<string>();

    void Start()
    {
        // 시작할 때 상태 텍스트를 초기화
        SetStatusIdle();
    }

    // "듣고 있습니다..." 상태로 변경
    public void SetStatusListening()
    {
        statusText.text = "듣고 있습니다...";
    }

    // "음성 인식 중..." 상태로 변경
    public void SetStatusProcessing()
    {
        statusText.text = "음성 인식 중...";
    }

    // 대기 상태로 변경 (텍스트 지우기)
    public void SetStatusIdle()
    {
        statusText.text = "";
    }

    // 새로운 히스토리 항목 추가
    public void AddToHistory(string newText)
    {
        if (historyContent == null || historyEntryPrefab == null) return;

        // 1. 히스토리 목록에 텍스트 추가
        commandHistory.Add(newText);

        // 2. 프리팹을 사용해 새로운 UI Text 오브젝트 생성
        GameObject newEntry = Instantiate(historyEntryPrefab, historyContent.transform);

        // 3. 생성된 오브젝트의 텍스트를 설정
        var textComponent = newEntry.GetComponent<TextMeshProUGUI>();
        if (textComponent != null)
        {
            textComponent.text = "> " + newText;
        }

        // (선택) 너무 많은 히스토리가 쌓이면 가장 오래된 것부터 지우는 로직 추가 가능
    }
}