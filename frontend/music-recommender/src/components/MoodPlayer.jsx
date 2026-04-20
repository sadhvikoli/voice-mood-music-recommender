import { useState, useRef } from "react";
import "./MoodPlayer.css";

const EMOTION_META = {
  happy:    { emoji: "😄", color: "#f9c74f", badge: "Upbeat" },
  sad:      { emoji: "😢", color: "#4895ef", badge: "Melancholic" },
  angry:    { emoji: "😤", color: "#e63946", badge: "Intense" },
  fearful:  { emoji: "😨", color: "#7209b7", badge: "Eerie" },
  disgust:  { emoji: "🤢", color: "#2dc653", badge: "Dark" },
  surprised:{ emoji: "😲", color: "#f77f00", badge: "Electric" },
  neutral:  { emoji: "😐", color: "#b3b3b3", badge: "Chill" },
  calm:     { emoji: "😌", color: "#90e0ef", badge: "Peaceful" },
};

const ART_COLORS = ["#e63946","#f77f00","#f9c74f","#2dc653","#4895ef","#7209b7","#90e0ef","#ff6b6b"];
const ART_NOTES  = ["🎵","🎶","🎸","🎹","🥁","🎷","🎺","🎻"];

export default function MoodPlayer() {
  const [isRecording, setIsRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [emotion, setEmotion] = useState(null);
  const [tracks, setTracks] = useState([]);
  const [secs, setSecs] = useState(0);
  const [error, setError] = useState("");

  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const analyserRef = useRef(null);
  const animRef = useRef(null);
  const barsRef = useRef(null);

  async function startRecording() {
    try {
      setError("");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioContextClass();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 128;
      ctx.createMediaStreamSource(stream).connect(analyser);

      analyserRef.current = {
        analyser,
        ctx,
        data: new Uint8Array(analyser.frequencyBinCount),
      };

      drawBars();

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";

      const recorder = new MediaRecorder(stream, { mimeType });
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      recorder.onstop = async () => {
        try {
          stream.getTracks().forEach((t) => t.stop());
          await ctx.close();

          const webmBlob = new Blob(chunksRef.current, { type: mimeType });
          const wavBlob = await convertToWav(webmBlob);
          await sendToBackend(wavBlob);
        } catch (err) {
          console.error(err);
          setError("Could not prepare audio for upload.");
        }
      };

      recorder.start();
      recorderRef.current = recorder;

      setIsRecording(true);
      setSecs(0);
      timerRef.current = setInterval(() => setSecs((s) => s + 1), 1000);
    } catch (err) {
      console.error(err);
      setError("Microphone access failed.");
    }
  }

  function stopRecording() {
    recorderRef.current?.stop();
    clearInterval(timerRef.current);
    cancelAnimationFrame(animRef.current);
    setIsRecording(false);
  }

  function drawBars() {
    if (!analyserRef.current) return;

    const { analyser, data } = analyserRef.current;
    analyser.getByteFrequencyData(data);

    if (barsRef.current) {
      const bars = barsRef.current.querySelectorAll(".wv-bar");
      bars.forEach((b, i) => {
        const v = data[Math.floor((i * data.length) / bars.length)] || 0;
        b.style.height = Math.max(4, (v / 255) * 40) + "px";
      });
    }

    animRef.current = requestAnimationFrame(drawBars);
  }

  async function sendToBackend(wavBlob) {
    setLoading(true);
    setError("");

    const fd = new FormData();
    fd.append("audio", wavBlob, "recording.wav");

    try {
      const res = await fetch("http://localhost:8000/recommend", {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Backend request failed");
      }

      const data = await res.json();
      setEmotion(data.emotion);
      setTracks(data.recommendations || []);
    } catch (err) {
      console.error(err);
      setError("Could not connect to backend.");
    } finally {
      setLoading(false);
    }
  }

  async function convertToWav(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioContextClass();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const wavArrayBuffer = audioBufferToWav(audioBuffer);
    await audioContext.close();
    return new Blob([wavArrayBuffer], { type: "audio/wav" });
  }

  function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1;
    const bitDepth = 16;

    let samples;
    if (numChannels === 2) {
      samples = interleave(buffer.getChannelData(0), buffer.getChannelData(1));
    } else {
      samples = buffer.getChannelData(0);
    }

    const blockAlign = (numChannels * bitDepth) / 8;
    const byteRate = sampleRate * blockAlign;
    const dataSize = samples.length * 2;
    const wavBuffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(wavBuffer);

    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(view, 36, "data");
    view.setUint32(40, dataSize, true);

    floatTo16BitPCM(view, 44, samples);
    return wavBuffer;
  }

  function interleave(left, right) {
    const result = new Float32Array(left.length + right.length);
    let index = 0;
    let inputIndex = 0;

    while (index < result.length) {
      result[index++] = left[inputIndex];
      result[index++] = right[inputIndex];
      inputIndex++;
    }

    return result;
  }

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  function floatTo16BitPCM(view, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, input[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
  }

  const fmt = (s) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

  const meta = emotion
    ? (EMOTION_META[emotion.toLowerCase()] || EMOTION_META.neutral)
    : null;

  return (
    <div className="sp">
      <div className="sp-header">
        <div className="sp-logo">● Mood-y</div>
        <h1 className="sp-title">How are you feeling?</h1>
        <p className="sp-sub">Record your voice — we'll match the music to your mood</p>
      </div>

      <div className="sp-body">
        <div className="recorder-zone">
          <div className="waveform" ref={barsRef}>
            {isRecording
              ? Array.from({ length: 36 }).map((_, i) => <div key={i} className="wv-bar" />)
              : <span className="hint">tap the mic to start</span>}
          </div>

          <button
            className={`mic-btn ${isRecording ? "recording" : ""}`}
            onClick={isRecording ? stopRecording : startRecording}
            disabled={loading}
          >
            {isRecording ? "⏹" : "🎙"}
          </button>

          {isRecording && <div className="timer">{fmt(secs)}</div>}
        </div>

        {loading && (
          <div className="loading-wrap">
            <div className="spin" />
            <p className="loading-text">Analyzing your vibe...</p>
          </div>
        )}

        {error && (
          <div style={{ color: "#ff8a8a", marginBottom: "1rem", textAlign: "center" }}>
            {error}
          </div>
        )}

        {meta && !loading && (
          <div className="emotion-banner">
            <div className="emotion-emoji">{meta.emoji}</div>
            <div>
              <div className="emotion-label">detected mood</div>
              <div className="emotion-name">
                {emotion.charAt(0).toUpperCase() + emotion.slice(1)}
              </div>
            </div>
            <span
              className="emotion-badge"
              style={{ background: meta.color + "25", color: meta.color }}
            >
              {meta.badge}
            </span>
          </div>
        )}

        {tracks.length > 0 && !loading && (
          <div className="tracks-section">
            <div className="section-label">Recommended for you</div>
            <div className="track-list">
              {tracks.map((t, i) => (
                <div key={i} className="track-row">
                  <div className="track-num">{i + 1}</div>
                  <div
                    className="track-art"
                    style={{ background: ART_COLORS[i % ART_COLORS.length] + "25" }}
                  >
                    {ART_NOTES[i % ART_NOTES.length]}
                  </div>
                  <div className="track-info">
                    <div className="track-name">{t.track_name}</div>
                    <div className="track-artist">{t.artists}</div>
                  </div>
                  <div className="track-genre">{t.track_genre}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}