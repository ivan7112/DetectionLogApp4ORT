import * as ort from "onnxruntime-web";

// Mismo orden que model.names en Python (print(model.names)).
// Sustituye estos nombres por las clases de tu clasificador de posición.
export const CLASS_NAMES = ["abajo", "arriba", "centro", "derecha", "izquierda"];
export const EYE_CLASS_NAMES=["abajo", "arriba", "centro", "derecha", "izquierda"];

export type Classification = {
    label: string;
    index: number;
    score: number;
    probs: number[]; // todas las probabilidades
};

export function classifyGazeDirection(output: ort.Tensor): Classification {
    const raw = output.data as Float32Array;
    let values = Array.from(raw);

    // Ultralytics exporta el softmax incluido, así que la suma ya es 1.
    // Si no lo fuera (logits), lo aplicamos aquí.
    const sum = values.reduce((a, b) => a + b, 0);
    const isProbability = Math.abs(sum - 1) < 0.01 && values.every((v) => v >= 0);

    if (!isProbability) {
        const max = Math.max(...values); // restar el máximo evita desbordamientos en exp()
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

    // Ultralytics exporta el softmax incluido, así que la suma ya es 1.
    // Si no lo fuera (logits), lo aplicamos aquí.
    const sum = values.reduce((a, b) => a + b, 0);
    const isProbability = Math.abs(sum - 1) < 0.01 && values.every((v) => v >= 0);

    if (!isProbability) {
        const max = Math.max(...values); // restar el máximo evita desbordamientos en exp()
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