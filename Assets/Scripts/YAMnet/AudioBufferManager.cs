using UnityEngine;
using System.Collections.Generic;

public class AudioBufferManager : MonoBehaviour
{
    [Header("Audio Settings")]
    private const int FREQUENCY = 16000; // Sampling rate (Hz)
    private const float BUFFER_TIME_SECONDS = 0.96f; // The duration of each audio buffer to process (in seconds)

    [Header("YAMNet Integration")]
    [Tooltip("Assign the GameObject that has the RunYamNet script responsible for YAMNet processing.")]
    public RunYamNet yamNetObject; // Reference to the RunYamNet script

    [Tooltip("Enable or disable YAMNet inference.")]
    public bool enableYamNet = true; // Flag to enable/disable YAMNet

    // Microphone-related variables
    private AudioClip _microphoneClip;
    private string _microphoneDevice;
    private int _lastSamplePosition;

    // Buffer for storing audio data before processing
    private List<float> _sampleBuffer = new List<float>();
    private int _samplesPerChunk;

    void Start()
    {
        // Check if the YAMNet object is assigned
        if (yamNetObject == null)
        {
            Debug.LogError("YAMNet object is not assigned! Please set the 'Yam Net Object' field in the Inspector.");
            // Depending on requirements, you might want to stop execution here.
        }
        
        if (Microphone.devices.Length == 0)
        {
            Debug.LogError("No microphone devices available.");
            return;
        }
        _microphoneDevice = Microphone.devices[0];

        _samplesPerChunk = (int)(FREQUENCY * BUFFER_TIME_SECONDS);

        // Start recording with a looping clip that's longer than our buffer time for stability.
        _microphoneClip = Microphone.Start(_microphoneDevice, true, 2, FREQUENCY);
        
        // Wait until the microphone has started recording
        while (!(Microphone.GetPosition(null) > 0)) { }
        
        Debug.Log("Microphone recording started. Settings: " + FREQUENCY + "Hz, Mono");
    }

    void Update()
    {
        if (_microphoneClip == null) return;

        // Collect new audio data
        int currentPosition = Microphone.GetPosition(_microphoneDevice);
        ReadSamples(currentPosition);
        _lastSamplePosition = currentPosition;

        // Check if there's at least one full chunk of data to process.
        // Using 'if' instead of 'while' prevents potential frame drops by processing only one chunk per frame.
        if (_sampleBuffer.Count >= _samplesPerChunk)
        {
            float[] chunkToProcess = new float[_samplesPerChunk];
            _sampleBuffer.CopyTo(0, chunkToProcess, 0, _samplesPerChunk);

            // Remove the processed chunk from the buffer
            _sampleBuffer.RemoveRange(0, _samplesPerChunk);
            
            // Call the audio processing function
            ProcessAudioBuffer(chunkToProcess);
        }
    }

    /// <summary>
    /// Reads new samples from the microphone clip and adds them to the internal buffer.
    /// </summary>
    private void ReadSamples(int currentPosition)
    {
        int sampleCount;
        if (currentPosition < _lastSamplePosition) // A loop (wrap-around) has occurred
        {
            sampleCount = (_microphoneClip.samples - _lastSamplePosition) + currentPosition;
        }
        else
        {
            sampleCount = currentPosition - _lastSamplePosition;
        }

        if (sampleCount > 0)
        {
            float[] newSamples = new float[sampleCount];
            // GetData does not read in a circular way, so we start from _lastSamplePosition.
            _microphoneClip.GetData(newSamples, _lastSamplePosition);
            _sampleBuffer.AddRange(newSamples);
        }
    }

    /// <summary>
    /// Passes the 0.96-second audio buffer to YAMNet for processing.
    /// </summary>
    private void ProcessAudioBuffer(float[] buffer)
    {
        // Call the function only if the YAMNet object is assigned and the feature is enabled.
        if (yamNetObject != null && enableYamNet)
        {
            // The 'buffer' is already 16000Hz, 0.96-second data, so we pass it directly.
            yamNetObject.AppendAudio(buffer, enableYamNet);
            Debug.Log("Passing audio buffer to YAMNet. Sample count: " + buffer.Length);
        }
        else
        {
            // Behavior when YAMNet is disabled (e.g., just logging)
            Debug.Log("Audio buffer created (YAMNet disabled). Sample count: " + buffer.Length);
        }
    }

    void OnDestroy()
    {
        // Stop the microphone when the object is destroyed
        if (_microphoneDevice != null && Microphone.IsRecording(_microphoneDevice))
        {
            Microphone.End(_microphoneDevice);
            Debug.Log("Stopping microphone recording.");
        }
    }
}