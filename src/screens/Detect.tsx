
import * as ort from "onnxruntime-web";
import Canvas from "../components/Canvas";
import { useEffect, useState, useRef } from "react";
import { videoToTensor} from "../utils/videototensor";
import processOutput from "../utils/processOutput";



export default function Detect(){

    const videoRef = useRef<HTMLVideoElement>(null);
    const [video, setVideo] = useState<HTMLVideoElement | null>(null);

    const [start, setStart]=useState(false);

    const startRef=useRef(start);
    const intervalRef=useRef(30);


    const frameRef=useRef(0);
    const busyRef=useRef(false);
    const sessionRef = useRef<ort.InferenceSession | null>(null);

    const bboxRef = useRef<ReturnType<typeof processOutput>>(null);



    const getSession=async  ()=>{
        if (!sessionRef.current) sessionRef.current=await ort.
            InferenceSession.create('/head_detector.onnx',{executionProviders: ["webgpu", "wasm"]});
        
        return sessionRef.current;
    }
    

    useEffect(() => {
        let stream: MediaStream | null = null;
        let cancelled = false;
        

        (async () => {
            if(start){
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

            }
        })();

        return () => {
        cancelled = true;
        stream?.getTracks().forEach((t) => t.stop());
        };
  }, [start]);

    useEffect(() => { startRef.current = start; }, [start]);
    

    const headDetectionInference=async  (video:HTMLVideoElement) =>{

        const {tensor, letterbox}=videoToTensor(video);
        const session=await getSession();

        const result=await session.run({[session.inputNames[0]]:tensor});

        const output=result[session.outputNames[0]];


        bboxRef.current=processOutput(output, letterbox); //tensor to bbox 

        //console.log(bboxRef.current);
        
        busyRef.current=false;
    }

    
    useEffect(()=>{
        let rafid:number;
        
        const loop=async ()=>{
            
        if (!startRef.current) frameRef.current=0;
        else frameRef.current=frameRef.current+1;
      
        if(!busyRef.current && frameRef.current >=intervalRef.current && startRef.current){
            busyRef.current=true;
            const v=videoRef.current;

            await headDetectionInference(v!);
            
            frameRef.current=0;
            busyRef.current=false;
        }
        rafid=requestAnimationFrame(loop);
        
    }
    rafid=requestAnimationFrame(loop);
    return ()=>cancelAnimationFrame(rafid); //Sin esto sigue el loop incluso al desmontar
    },[]);

  //<video> es para decodificar la información de 

    return (
        <>
        <p>WIP as of now</p>

        <div>
        <button onClick={()=>{
            setStart(true);
        }}>Start Now!</button>
        </div>

        <div>
            <button onClick={()=>{
                setStart(false);
            }}>Close Camera</button>
        </div>

        <div>
            <label htmlFor="frames">Catch every x frames:  </label>
            <input id="frames" type="text" defaultValue={30} onChange={(event)=>{
                intervalRef.current=parseInt(event.target.value);
            }}/>

        </div>

        {start && (

        <div>
        <video ref={videoRef} autoPlay playsInline muted style={{display:'none'}} />
        <Canvas width={640} height={480} source={video!} bbox={bboxRef!} />
        </div>
        )}

        <button onClick={()=>{
            console.log(frameRef.current);
        }}>Log frameRef</button>
        </>
    )
}