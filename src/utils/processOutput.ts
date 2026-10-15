
import type { Letterbox } from "./videototensor";
import * as ort from "onnxruntime-web";


export type BBox = { x: number; y: number; w: number; h: number; score: number };
const CONF_THRESHOLD = 0.25;


const processOutput = (output: ort.Tensor, lb: Letterbox): BBox | null => {
    const data = output.data as Float32Array;
    const [, channels, n] = output.dims; // [1, 4 + clases, candidatos]

    // Layout channels-first: el valor del canal c, candidato i, está en data[c * n + i]
    let bestIdx = -1;
    let bestScore = CONF_THRESHOLD;

    for (let i = 0; i < n; i++) {
        // con varias clases, el score es el máximo de los canales 4..channels-1
        let score = 0;
        for (let c = 4; c < channels; c++) {
            score = Math.max(score, data[c * n + i]);
        }
        if (score > bestScore) {
            bestScore = score;
            bestIdx = i;
        }
    }

    if (bestIdx === -1) return null; // ninguna detección supera el umbral

    const cx = data[0 * n + bestIdx]; //dimensiones bbox
    const cy = data[1 * n + bestIdx];
    const w  = data[2 * n + bestIdx];
    const h  = data[3 * n + bestIdx];

    const res={
        x: (cx - w / 2 - lb.padX) / lb.scale,
        y: (cy - h / 2 - lb.padY) / lb.scale,
        w: w / lb.scale,
        h: h / lb.scale,
        score: bestScore,
    }
    
    return res;
};

export default processOutput;
