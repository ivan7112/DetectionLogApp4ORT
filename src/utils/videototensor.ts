import * as ort from "onnxruntime-web";

const SIZE = 640; // tamaño de entrada del modelo

// Se crean una vez y se reutilizan (crear un canvas por frame es caro)
const offscreen = document.createElement("canvas");
offscreen.width = SIZE;
offscreen.height = SIZE;
const ctx = offscreen.getContext("2d", { willReadFrequently: true })!;
const floatData = new Float32Array(3 * SIZE * SIZE); //3 valores por pixel

export type Letterbox = { scale: number; padX: number; padY: number };

export function videoToTensor(video: HTMLVideoElement): { tensor: ort.Tensor; letterbox: Letterbox } {
    const vw = video.videoWidth;
    const vh = video.videoHeight;

    // Escala manteniendo proporción y centrado
    const scale = Math.min(SIZE / vw, SIZE / vh);
    const w = Math.round(vw * scale);
    const h = Math.round(vh * scale);
    const padX = Math.floor((SIZE - w) / 2);
    const padY = Math.floor((SIZE - h) / 2);

    // Fondo gris + frame
    ctx.fillStyle = "rgb(114,114,114)";
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.drawImage(video, padX, padY, w, h);

    // RGBA intercalado -> NCHW planar, normalizado
    const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
    const plane = SIZE * SIZE;

    for (let i = 0; i < plane; i++) {
        floatData[i] = data[i * 4]     / 255; // R
        floatData[i + plane] = data[i * 4 + 1] / 255; // G
        floatData[i + 2 * plane] = data[i * 4 + 2] / 255; // B
    }

    const tensor = new ort.Tensor("float32", floatData, [1, 3, SIZE, SIZE]);
    return { tensor, letterbox: { scale, padX, padY } };
}