// VoiceCommandSO.cs
using UnityEngine;
using System.Collections.Generic;
using System;

[CreateAssetMenu(fileName = "New Voice Command", menuName = "Voice/Command")]
public class VoiceCommandSO : ScriptableObject
{
    [Tooltip("나중에 AI가 사용할 키워드 목록")]
    public List<string> keywords = new List<string>();


    public event Action onCommandRecognized;

    public void Invoke()
    {
        Debug.Log(this.name + " 이벤트가 호출되었습니다!");
        onCommandRecognized?.Invoke();
    }
}