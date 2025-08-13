// 이 스크립트는 어떤 오브젝트에도 붙이지 않습니다.
using UnityEngine.Events;
using System;

// Inspector 창에 리스트 형태로 보여주기 위해 [Serializable] 속성을 추가합니다.
[Serializable]
public class CommandEventMapping
{
    public VoiceCommandSO command;
    public UnityEvent onCommandHeard;
}