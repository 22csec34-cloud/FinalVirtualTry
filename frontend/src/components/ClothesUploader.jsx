import { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";

export default function ClothesUploader() {
  const [clothes, setClothes] = useState([]);
  const [prompt, setPrompt] = useState("");

  const onDrop = useCallback((acceptedFiles) => {
    const newImages = acceptedFiles.map((file) =>
      Object.assign(file, {
        preview: URL.createObjectURL(file),
        id: crypto.randomUUID(),
      })
    );
    setClothes((prev) => [...prev, ...newImages]);
  }, []);

  const removeCloth = (id) => {
    setClothes((prev) => prev.filter((item) => item.id !== id));
  };

  useEffect(() => {
    return () => clothes.forEach((f) => URL.revokeObjectURL(f.preview));
  }, [clothes]);

  const { getRootProps, getInputProps } = useDropzone({
    accept: { "image/*": [] },
    onDrop,
  });

  return (
    <div className="card">
      <h2>Clothes Upload</h2>

      <div {...getRootProps()} className="dropzone">
        <input {...getInputProps()} />
        <p>Drag & drop clothes here or click</p>
      </div>

      <div className="clothes-grid">
        {clothes.map((file) => (
          <div key={file.id} className="cloth-wrapper">
            <img src={file.preview} className="cloth-img" />
            <button
              className="delete-btn"
              onClick={() => removeCloth(file.id)}
            >
              ✕
            </button>
          </div>
        ))}
      </div>

      <textarea
        className="prompt-box"
        placeholder="Describe your dream outfit..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />

      <button className="generate-btn">
        ✨ Generate 360° AI Design
      </button>
    </div>
  );
}