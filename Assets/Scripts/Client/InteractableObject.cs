using UnityEngine;

public class InteractableObject : MonoBehaviour
{
    // 초기 상태 저장을 위한 변수들
    private Vector3 initialScale;
    private Quaternion initialRotation;
    private bool isRotating = false;

    void Start()
    {
        // 앱 시작 시 초기 상태를 저장
        initialScale = transform.localScale;
        initialRotation = transform.rotation;
    }

    void Update()
    {
        // isRotating이 true일 때만 회전
        if (isRotating)
        {
            transform.Rotate(Vector3.up, 50f * Time.deltaTime);
        }
    }

    // --- 음성 명령과 연결될 public 함수들 ---

    public void ZoomIn()
    {
        transform.localScale *= 1.2f;
        Debug.Log("명령: 확대");
    }

    public void ZoomOut()
    {
        transform.localScale /= 1.2f;
        Debug.Log("명령: 축소");
    }

    public void StartRotation()
    {
        isRotating = true;
        Debug.Log("명령: 회전 시작");
    }

    public void StopRotation()
    {
        isRotating = false;
        Debug.Log("명령: 회전 정지");
    }

    public void ResetObject()
    {
        transform.localScale = initialScale;
        transform.rotation = initialRotation;
        isRotating = false;
        Debug.Log("명령: 원래대로");
    }
}