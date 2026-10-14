
import * as ort from "onnxruntime-web";
import Canvas from "../components/Canvas";
import { useEffect, useState, useRef } from "react";


export default function Detect(){

    const videoRef = useRef<HTMLVideoElement>(null);
    const [video, setVideo] = useState<HTMLVideoElement | null>(null);

    useEffect(() => {
        let stream: MediaStream | null = null;
        let cancelled = false;

        (async () => {
        const s = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (cancelled) {
            s.getTracks().forEach((t) => t.stop());
            return;
        }
        stream = s;
        const v = videoRef.current!;
        v.srcObject = s;
        await v.play();
        setVideo(v); // ahora Canvas empieza a dibujar
        })();

        return () => {
        cancelled = true;
        stream?.getTracks().forEach((t) => t.stop());
        };
  }, []);

    const headDetectionInference=async  (tensor:ort.Tensor) =>{

        

        

        const session=await ort.InferenceSession.create('../yolo-models/detectors/head_detector.onnx');
        
        const feeds={a:tensor};

        const result=await session.run(feeds);

        const output=result[session.outputNames[0]];

        //Sacar bbox

    }



    return (
        <>
        <p>WIP as of now</p>
        <Canvas width={500} height={600} source={video!} />
        </>
    )
}