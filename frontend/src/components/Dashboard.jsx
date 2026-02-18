import { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";

export default function TryOnInput() {
  const [userImage, setUserImage] = useState(null);
  const [clothes, setClothes] = useState([]);
  const [prompt, setPrompt] = useState("");

  const handleUserImage = (e) => {
    const file = e.target.files[0];
    if (file) setUserImage(URL.createObjectURL(file));
  };

  const onDrop = useCallback((acceptedFiles) => {
    const images = acceptedFiles.map((file) =>
      Object.assign(file, {
        preview: URL.createObjectURL(file),
        id: crypto.randomUUID(),
      })
    );
    setClothes((prev) => [...prev, ...images]);
  }, []);

  const removeCloth = (id) => {
    setClothes((prev) => prev.filter((c) => c.id !== id));
  };

  useEffect(() => {
    return () => clothes.forEach((f) => URL.revokeObjectURL(f.preview));
  }, [clothes]);

  const { getRootProps, getInputProps } = useDropzone({
    accept: { "image/*": [] },
    onDrop,
  });

  return (
    <div className="card tryon-card">
      <h2>Virtual Try-On</h2>

      {/* USER IMAGE */}
      <label className="upload-box">
        {userImage ? (
          <img src={userImage} className="preview-img" />
        ) : (
          <p>Upload User Image</p>
        )}
        <input type="file" hidden onChange={handleUserImage} />
      </label>

      {/* CLOTHES */}
      <div {...getRootProps()} className="dropzone">
        <input {...getInputProps()} />
        <p>Drag & drop clothes here</p>
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

      {/* PROMPT */}
      <textarea
        className="prompt-box"
        placeholder="Describe your dream outfit..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />

      <button className="generate-btn">
        ✨ Generate AI Try-On
      </button>
    </div>
  );
}
