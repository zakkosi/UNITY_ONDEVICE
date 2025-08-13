using UnityEngine;

public class UIManager : MonoBehaviour
{
    // Inspector에서 InfoPopup 패널을 연결해 줄 변수
    public GameObject infoPopup;

    public void ShowInfoPopup()
    {
        if (infoPopup != null)
        {
            infoPopup.SetActive(true);
            Debug.Log("명령: 설명창 열기");
        }
    }

    public void HideInfoPopup()
    {
        if (infoPopup != null)
        {
            infoPopup.SetActive(false);
            Debug.Log("명령: 설명창 닫기");
        }
    }
}