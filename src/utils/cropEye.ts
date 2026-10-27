import * as ort from "onnxruntime-web/webgpu";
import type { BBox } from "./processOutput";

const EYE_SIZE = 224; //Eye classificator's imgsz

//Proportions regarding head's bbox (0 to 1). Adjust manually
const EYE_CENTER_X = 0.65; // 0.65 = Left eye (Right in the image)
const EYE_CENTER_Y = 0.40; //Eye's height from bbox's top
const EYE_SIDE = 0.35;     //side of squared crop, as a fraction of bbox's width

// Canvas & buffer are created only once
const canvas = document.createElement("canvas");
canvas.width = EYE_SIZE;
canvas.height = EYE_SIZE;
const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
const floatData = new Float32Array(3 * EYE_SIZE * EYE_SIZE);
const plane = EYE_SIZE * EYE_SIZE;

export function eyeCropToTensor(video: HTMLVideoElement, box: BBox): ort.Tensor {
    const vw = video.videoWidth;
    const vh = video.videoHeight;

    // Square centered in eye's estimated position
    const side = Math.min(box.w * EYE_SIDE, vw, vh);
    const cx = box.x + box.w * EYE_CENTER_X;
    const cy = box.y + box.h * EYE_CENTER_Y;

    // Moves crop if its bounds are outside the canvas
    const sx = Math.min(Math.max(cx - side / 2, 0), vw - side);
    const sy = Math.min(Math.max(cy - side / 2, 0), vh - side);

    ctx.drawImage(video, sx, sy, side, side, 0, 0, EYE_SIZE, EYE_SIZE);

    // Intercalated RGB -> plannar NCHW, normalized in [0, 1]
    const { data } = ctx.getImageData(0, 0, EYE_SIZE, EYE_SIZE);
    for (let i = 0; i < plane; i++) {
        floatData[i]             = data[i * 4]     / 255;
        floatData[i + plane]     = data[i * 4 + 1] / 255;
        floatData[i + 2 * plane] = data[i * 4 + 2] / 255;
    }

    return new ort.Tensor("float32", floatData, [1, 3, EYE_SIZE, EYE_SIZE]);
}

// for debugging, ignore
export const eyeDebugCanvas = canvas;