import { useState } from "react";

export default function UserImage() {
  const [image, setImage] = useState(null);

  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (file) setImage(URL.createObjectURL(file));
  };

  return (
    <div className="card">
      <h2>User Image</h2>

      <label className="upload-box">
        {image ? (
          <img src={image} className="preview-img" />
        ) : (
          <p>Click to upload user photo</p>
        )}
        <input type="file" hidden onChange={handleUpload} />
      </label>
    </div>
  );
}
