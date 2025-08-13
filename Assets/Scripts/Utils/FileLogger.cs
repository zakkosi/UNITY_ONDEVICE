using UnityEngine;
using System.IO;
using System;

public static class FileLogger
{
    private static readonly string logFilePath;
    private static readonly object fileLock = new object();

    static FileLogger()
    {
        try
        {

            string logDirectory = Path.Combine(Application.persistentDataPath, "logs");

            // 로그 디렉토리가 없으면 생성
            if (!Directory.Exists(logDirectory))
            {
                Directory.CreateDirectory(logDirectory);
            }

            logFilePath = Path.Combine(logDirectory, $"log_{DateTime.Now:yyyy-MM-dd}.txt");


            Log("--- Application Started ---");
        }
        catch (Exception e)
        {
            Debug.LogError($"[FileLogger] Failed to initialize: {e.Message}");
        }
    }

    public static void Log(string message)
    {

        Debug.Log(message);


        if (string.IsNullOrEmpty(logFilePath) || string.IsNullOrEmpty(message))
        {
            return;
        }

        try
        {

            lock (fileLock)
            {

                File.AppendAllText(logFilePath, $"[{DateTime.Now:HH:mm:ss.fff}] {message}\n");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[FileLogger] Failed to write to log file: {e.Message}");
        }
    }
}