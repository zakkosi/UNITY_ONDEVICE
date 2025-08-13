// VoiceCommandListener.cs
using UnityEngine;
using UnityEngine.Events;

public class VoiceCommandListener : MonoBehaviour
{
    [Tooltip("이 리스너가 귀 기울일 명령어 에셋")]
    public VoiceCommandSO commandToListenFor;

    [Tooltip("명령어가 들렸을 때, 이 게임오브젝트에서 실행할 기능들")]
    public UnityEvent onCommandHeard;




    // 오브젝트가 활성화될 때 '듣기' 시작
    private void OnEnable()
    {
        if (commandToListenFor != null)
        {
            commandToListenFor.onCommandRecognized += OnCommandHeard;
        }
    }

    // 오브젝트가 비활성화될 때 '듣기' 중지
    private void OnDisable()
    {
        if (commandToListenFor != null)
        {
            commandToListenFor.onCommandRecognized -= OnCommandHeard;
        }
    }

    // 신호를 들었을 때 실행될 함수
    private void OnCommandHeard()
    {
        // Inspector에 연결된 UnityEvent를 실행
        onCommandHeard.Invoke();
    }
}