import * as ort from "onnxruntime-web";


//Same order as model.names in Python 
//Replace this names with your own classificator's classes
export const CLASS_NAMES = ["down", "up", "center", "right", "left"];
export const EYE_CLASS_NAMES=["down", "up", "center", "right", "left"];

export type Classification = {
    label: string;
    index: number;
    score: number;
    probs: number[]; // All probabilities
};

export function classifyGazeDirection(output: ort.Tensor): Classification {
    const raw = output.data as Float32Array;
    let values = Array.from(raw);

    //Ultralytics export includes softmax, so total sum it's already 1
    //In case it wasn't (logits), we apply it here
    const sum = values.reduce((a, b) => a + b, 0);
    const isProbability = Math.abs(sum - 1) < 0.01 && values.every((v) => v >= 0);

    if (!isProbability) { //softmax, to convert to probabilities
        const max = Math.max(...values); //By substracting max we avoid overflows on exp()
        const exps = values.map((v) => Math.exp(v - max));
        const total = exps.reduce((a, b) => a + b, 0);
        values = exps.map((e) => e / total);
    }

    let best = 0;
    for (let i = 1; i < values.length; i++) {
        if (values[i] > values[best]) best = i;
    }

    return {
        label: EYE_CLASS_NAMES[best] ?? `clase ${best}`,
        index: best,
        score: values[best],
        probs: values,
    };
}


export function classifyHeadPosition(output: ort.Tensor): Classification {
    const raw = output.data as Float32Array;
    let values = Array.from(raw);

    
    const sum = values.reduce((a, b) => a + b, 0);
    const isProbability = Math.abs(sum - 1) < 0.01 && values.every((v) => v >= 0);

    if (!isProbability) { 
        const max = Math.max(...values); 
        const exps = values.map((v) => Math.exp(v - max));
        const total = exps.reduce((a, b) => a + b, 0);
        values = exps.map((e) => e / total);
    }

    let best = 0;
    for (let i = 1; i < values.length; i++) {
        if (values[i] > values[best]) best = i;
    }

    return {
        label: CLASS_NAMES[best] ?? `clase ${best}`,
        index: best,
        score: values[best],
        probs: values,
    };
}