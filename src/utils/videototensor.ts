import * as ort from "onnxruntime-web/webgpu";

export type Letterbox = { scale: number; padX: number; padY: number };

export function createVideoPreprocessor(size: number) {
    // A canvas and buffer per size, only created once
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    const floatData = new Float32Array(3 * size * size);
    const plane = size * size;

    return (video: HTMLVideoElement): { tensor: ort.Tensor; letterbox: Letterbox } => {
        const vw = video.videoWidth;
        const vh = video.videoHeight;

        //Scalling preserving proportion and center
        const scale = Math.min(size / vw, size / vh);
        const w = Math.round(vw * scale);
        const h = Math.round(vh * scale);
        const padX = Math.floor((size - w) / 2);
        const padY = Math.floor((size - h) / 2);

        ctx.fillStyle = "rgb(114,114,114)";
        ctx.fillRect(0, 0, size, size);
        ctx.drawImage(video, padX, padY, w, h);

        const { data } = ctx.getImageData(0, 0, size, size);
        for (let i = 0; i < plane; i++) {
            floatData[i]             = data[i * 4]     / 255; //R
            floatData[i + plane]     = data[i * 4 + 1] / 255; //G
            floatData[i + 2 * plane] = data[i * 4 + 2] / 255; //B
        }

        const tensor = new ort.Tensor("float32", floatData, [1, 3, size, size]); //1 photo, 3 channels, mxn
        return { tensor, letterbox: { scale, padX, padY } };
    };
}

export const videoToTensor = createVideoPreprocessor(640);               // detector
export const videoToHeadPoseTensor = createVideoPreprocessor(384);       // head classificator