import './css/App.css';
import React, { useRef, useEffect, useState } from 'react';
import apiRequest from '../api/apirequest';

interface ImageDimensions {
  height: number;
  width: number;
}

function Pokedex() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imageHeight, setImageHeight] = useState<number>(0);
  const [imageWidth, setImageWidth] = useState<number>(0);

  useEffect(() => {
    const selectPhoto = (e: Event) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        setSelectedFile(file);
        readFile(file);
      } else {
        console.error('No file selected');
      }
    }

    if (fileInputRef.current) {
      fileInputRef.current.addEventListener('change', selectPhoto);
    }

    uploadImage(selectedFile);

    return () => {
      if (fileInputRef.current) {
        fileInputRef.current.removeEventListener('change', selectPhoto);
      }
    };
  });

  const onUploadPhoto = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  }

  const readFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e: ProgressEvent<FileReader>) => {
      const dataURL = e.target?.result as string;
      getImageDimensions(dataURL);
    }
    reader.readAsDataURL(file);
  }

  const getImageDimensions = (dataURL: string) => {
    const img = new Image();
    img.onload = () => {
      const height = img.naturalHeight;
      const width = img.naturalWidth;
      setImageHeight(height);
      setImageWidth(width);
    };
    img.src = dataURL;
  }

  const uploadImage = async (newImage: File | null) => {
    if (!newImage) return;
    const metadata = new FormData();
    metadata.append("file", newImage);
    metadata.append("height", imageHeight.toString());
    metadata.append("width", imageWidth.toString());
    await apiRequest("POST", "/upload", { metadata: metadata });
    setSelectedFile(null);
  }

  // TODO complete JSX
  return (
    <div id="Pokedex">
      <div id="left_block">
        <div id="camera">
          <div id="outer_circle"></div>
          <div id="inner_circle"></div>
        </div>
      </div>
      <div id="spine">
        <div id="spine_upper"></div>
        <div id="spine_lower"></div>
      </div>
      <div id="right_block">
        <div id="right_block_before_corner"></div>
        <div id="right_triangle_corner">
          <div id="triangle_corner"></div>
          <div id="triangle_corner_body"></div>
          <form id="upload_photo_form" encType="multipart/form-data">
            <input type="file" name="file" id="fileInput" ref={fileInputRef} hidden />
          </form>
          <button id="upload_photo" onClick={onUploadPhoto}>Take Photo</button>
        </div>
        <div id="right_block_after_corner"></div>
      </div>
    </div>
  );
}

export default Pokedex;
