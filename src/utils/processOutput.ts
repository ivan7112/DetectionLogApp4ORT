
import type { Letterbox } from "./videototensor";
import * as ort from "onnxruntime-web";


export type BBox = { x: number; y: number; w: number; h: number; score: number
    , classification: string };
const CONF_THRESHOLD = 0.25;

const handClasses:Record<number,string>={0: 'grabbing', 1: 'grip', 2: 'holy', 3: 'point', 4: 'call', 5: 'three3', 6: 'timeout', 7: 'xsign', 8: 'hand_heart', 9: 'hand_heart2', 10: 'little_finger', 11: 'middle_finger', 12: 'take_picture', 13: 'dislike', 14: 'fist', 15: 'four', 16: 'like', 17: 'mute', 18: 'ok', 19: 'one', 20: 'palm', 21: 'peace', 22: 'peace_inverted', 23: 'rock', 24: 'stop', 25: 'stop_inverted', 26: 'three', 27: 'three2', 28: 'two_up', 29: 'two_up_inverted', 30: 'three_gun', 31: 'thumb_index', 32: 'thumb_index2', 33: 'no_gesture'};
const headClasses:Record<number,string>={0: 'head', 1: 'hyy', 2: 'zzd'}

const processOutput = (output: ort.Tensor, lb: Letterbox, classes:Record<number,string>): BBox | null => {
    const data = output.data as Float32Array;
    const [, channels, n] = output.dims; // [1 image, 4 + classes channels, n candidates]

    // Layout channels-first: channel c value, candidate i, is at data[c * n + i]
    let bestIdx = -1;
    let bestScore = CONF_THRESHOLD;
    let final_class="";
    let chosenClass=0;
    for (let i = 0; i < n; i++) {
        let score = 0;
        for (let c = 4; c < channels; c++) { //each candidate gets its score for all classes
            score = Math.max(score, data[c * n + i]);

            if (score > bestScore) {
            bestScore = score;
            bestIdx = i;
            chosenClass=c-4;
            }     
        }
        
        
        
    }
    final_class=classes[chosenClass];
        

    if (bestIdx === -1) return null; // No detection reachs the umbral

    const cx = data[0 * n + bestIdx]; //bbox bounds
    const cy = data[1 * n + bestIdx];
    const w  = data[2 * n + bestIdx];
    const h  = data[3 * n + bestIdx];

    const res:BBox={
        x: (cx - w / 2 - lb.padX) / lb.scale,
        y: (cy - h / 2 - lb.padY) / lb.scale,
        w: w / lb.scale,
        h: h / lb.scale,
        score: bestScore,
        classification:final_class
    }
    
    return res;
};


const processHandGesture= (output: ort.Tensor, lb: Letterbox)=>{

    return processOutput(output, lb, handClasses);
}

const processHead= (output: ort.Tensor, lb: Letterbox)=>{

    return processOutput(output, lb, headClasses);
}

export  {processHandGesture};
export {processHead};
