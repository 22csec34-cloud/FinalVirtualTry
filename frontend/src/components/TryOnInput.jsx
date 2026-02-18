import { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";

export default function TryOnInput() {
  const [images, setImages] = useState([]);
  const [prompt, setPrompt] = useState("");

  const onDrop = useCallback((acceptedFiles) => {
    const files = acceptedFiles.map((file) =>
      Object.assign(file, {
        preview: URL.createObjectURL(file),
        id: crypto.randomUUID(),
      })
    );
    setImages((prev) => [...prev, ...files]);
  }, []);

  const removeImage = (id) => {
    setImages((prev) => prev.filter((img) => img.id !== id));
  };

  useEffect(() => {
    return () => images.forEach((f) => URL.revokeObjectURL(f.preview));
  }, [images]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "image/*": [] },
    multiple: true,
    onDrop,
  });

  return (
    <div className="card tryon-card">
      <h2>Virtual Try-On</h2>

      {/* DROPZONE */}
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? "active" : ""}`}
      >
        <input {...getInputProps()} />
        <p>
          Drag & drop <b>user image</b> and <b>clothes</b>
        </p>
        <span>Click to upload</span>
      </div>

      {/* PREVIEW GRID */}
      {images.length > 0 && (
        <div className="clothes-grid">
          {images.map((file, index) => (
            <div key={file.id} className="cloth-wrapper">
              <img src={file.preview} className="cloth-img" />

              {/* USER BADGE */}
              {index === 0 && (
                <span className="user-badge">USER</span>
              )}

              {/* DELETE BUTTON */}
              <button
                className="delete-btn"
                onClick={() => removeImage(file.id)}
                title={
                  index === 0
                    ? "Remove user image"
                    : "Remove clothing image"
                }
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}

      {/* PROMPT */}
      <textarea
        className="prompt-box"
        placeholder="Describe how the outfit should look..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />

      <button
        className="generate-btn"
        disabled={images.length < 2}
      >
        ✨ Generate AI Try-On
      </button>
    </div>
  );
}