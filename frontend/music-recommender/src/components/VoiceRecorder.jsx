import { useState, useRef } from "react";

export default function VoiceRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [recordings, setRecordings] = useState([]);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;
    chunksRef.current = [];

    mediaRecorder.ondataavailable = (e) => chunksRef.current.push(e.data);

    mediaRecorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      const url = URL.createObjectURL(blob);
      setRecordings((prev) => [...prev, { blob, url, name: `recording_${prev.length + 1}.wav` }]);
      stream.getTracks().forEach((t) => t.stop());
    };

    mediaRecorder.start();
    setIsRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
  };

  const sendToBackend = async (rec) => {
    const formData = new FormData();
    formData.append("audio", rec.blob, rec.name); // backend sees it as a file

    const res = await fetch("http://localhost:8000/upload-audio", {
      method: "POST",
      body: formData,
    });

    if (res.ok) alert("Uploaded!");
    else alert("Upload failed");
  };

  return (
    <div>
      {!isRecording ? (
        <button onClick={startRecording}>🎙 Start Recording</button>
      ) : (
        <button onClick={stopRecording}>⏹ Stop</button>
      )}

      {recordings.map((rec, i) => (
        <div key={i}>
          <audio src={rec.url} controls />
          <a href={rec.url} download={rec.name}>Download .wav</a>
          <button onClick={() => sendToBackend(rec)}>Send to Backend</button>
        </div>
      ))}
    </div>
  );
}