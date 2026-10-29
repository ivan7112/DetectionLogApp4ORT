import * as ort from "onnxruntime-web";


//Same order as model.names in Python 
//Replace this names with your own classificator's classes
export const HEAD_POSE_NAMES = ["down", "up", "center", "right", "left"];
export const EYE_CLASS_NAMES=["down", "up", "center", "right", "left"];

export type Classification = {
    label: string;
    index: number;
    score: number;
    probs: number[]; // All probabilities
};



const classification= (output:ort.Tensor, classes:Array<string>): Classification =>{

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
        label: classes[best] ?? `clase ${best}`,
        index: best,
        score: values[best],
        probs: values,
    };

}

export const classifyHeadPosition = (output:ort.Tensor) =>{
    return classification(output,HEAD_POSE_NAMES);
}

export const classifyGazeDirection = (output:ort.Tensor) =>{
    return classification(output,EYE_CLASS_NAMES);
}