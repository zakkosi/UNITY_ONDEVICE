// 이 스크립트를 TargetObject에 딱 한 번만 붙여줍니다.
using UnityEngine;
using System.Collections.Generic;

public class ObjectCommandRouter : MonoBehaviour
{
    // 위에서 만든 데이터 상자를 리스트(목록) 형태로 가집니다.
    public List<CommandEventMapping> commandMappings;

    private void OnEnable()
    {
        // 리스트에 있는 모든 '명령-기능' 짝에 대해 리스너를 등록합니다.
        foreach (var mapping in commandMappings)
        {
            if (mapping.command != null)
            {
                // 명령(SO)의 이벤트가 발생하면, 이 mapping에 연결된 UnityEvent를 실행하도록 등록
                mapping.command.onCommandRecognized += mapping.onCommandHeard.Invoke;
            }
        }
    }

    private void OnDisable()
    {
        // 리스너를 깨끗하게 해제합니다.
        foreach (var mapping in commandMappings)
        {
            if (mapping.command != null)
            {
                mapping.command.onCommandRecognized -= mapping.onCommandHeard.Invoke;
            }
        }
    }
}