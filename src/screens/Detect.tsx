
import * as ort from "onnxruntime-web";
import Canvas from "../components/Canvas";
import { useEffect, useState, useRef } from "react";
import { videoToTensor, videoToHeadPoseTensor} from "../utils/videototensor";
import processOutput from "../utils/processOutput";
import { classifyGazeDirection, classifyHeadPosition, type Classification } from "../utils/classifyHeadPosition";
import { eyeCropToTensor } from "../utils/cropEye";
import createCsv from "../utils/createCsv";



export default function Detect(){

    interface CsvRow{
        HH:string,
        MM:string,
        SS:string,
        head_position:string,
        gaze_direction:string
    }

    const videoRef = useRef<HTMLVideoElement>(null);
    const [video, setVideo] = useState<HTMLVideoElement | null>(null);

    const [start, setStart]=useState(false);
    const [headPosition,setHeadPosition]=useState("");
    const [gazeDirection, setGazeDirection]=useState("");

    const startRef=useRef(start);
    const intervalRef=useRef(10);
    const frameRef=useRef(0);
    const busyRef=useRef(false);
    const sessionRef = useRef<ort.InferenceSession | null>(null);
    const sessionHeadPositionRef=useRef<ort.InferenceSession | null>(null);
    const sessionGazeDirectionRef=useRef<ort.InferenceSession | null>(null);

    const bboxRef = useRef<ReturnType<typeof processOutput>>(null);

    //Líneas para el CSV. Tendrán formato HH,MM,SS,head_position,gaze_direction
    const csvRows = useRef<Array<CsvRow>>([]);



    const getSession=async  ()=>{
        if (!sessionRef.current) sessionRef.current=await ort.
            InferenceSession.create('/head_detector.onnx',{executionProviders: ["webgpu", "wasm"]});
        
        return sessionRef.current;
    }

    const getSessionHeadPosition=async  ()=>{
        if (!sessionHeadPositionRef.current) sessionHeadPositionRef.current=await ort.
            InferenceSession.create('/head_classificator_v2_alt.onnx',
                {executionProviders: ["webgpu", "wasm"]});
        
        return sessionHeadPositionRef.current;
    }

    const getSessiongazeDirection=async  ()=>{
        if (!sessionGazeDirectionRef.current) sessionGazeDirectionRef.current=await ort.
            InferenceSession.create('/gaze_direction_classificator.onnx',{executionProviders: ["webgpu", "wasm"]});
        
        return sessionGazeDirectionRef.current;
    }
    

    useEffect(() => {
        let stream: MediaStream | null = null;
        let cancelled = false;
        

        (async () => { //abrir cámara. 
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
        //Va guardando las detecciones en csvRows hasta que se cierre la cámara

        let rowCsv:CsvRow={
            HH:"",
            MM:"",
            SS:"",
            head_position:"",
            gaze_direction:""

        };

        const {tensor, letterbox}=videoToTensor(video);
        const session=await getSession();

        const result=await session.run({[session.inputNames[0]]:tensor});

        const output=result[session.outputNames[0]];


        bboxRef.current=processOutput(output, letterbox); //tensor to bbox 

        //Clasificación si se detecta una cabeza
        if (bboxRef.current){


            const {tensor}=videoToHeadPoseTensor(video);
            const sessionHeadPose=await getSessionHeadPosition();
            const resultHeadPosition=await sessionHeadPose.run({[sessionHeadPose.inputNames[0]]:tensor})
            const output2=resultHeadPosition[sessionHeadPose.outputNames[0]];

            const headPoseClass:Classification=classifyHeadPosition(output2);

            setHeadPosition(headPoseClass.label);

            const date= new Date();
            const hour=date.getHours();
            const minute=date.getMinutes();
            const second=date.getSeconds();

            const headPosition=headPoseClass.label;

            rowCsv={
                HH:hour.toString(),
                MM:minute.toString(),
                SS:second.toString(),
                head_position:headPosition,
                gaze_direction:""
            }

            if (headPosition==="centro"){
                const tensor=eyeCropToTensor(video, bboxRef.current);

                const sessionGazeDirection=await  getSessiongazeDirection();
                const resultGazeDirection= await sessionGazeDirection.run({[ sessionGazeDirection.inputNames[0]]:tensor});
                const output3= resultGazeDirection[sessionGazeDirection.outputNames[0]];

                const gazeDirectionClass=classifyGazeDirection(output3);
                setGazeDirection(gazeDirectionClass.label);

                rowCsv={...rowCsv, gaze_direction:gazeDirectionClass.label};
            }

            csvRows.current.push(rowCsv);

            
        }
        
        busyRef.current=false;
    }

    
    useEffect(()=>{ //Loop principal
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
                createCsv(csvRows.current);
                setStart(false);
                csvRows.current=[]; //Reseteamos CSV
            }}>Close Camera</button>
        </div>

        <div>
            <label htmlFor="frames">Catch every x frames:  </label>
            <input id="frames" type="text" defaultValue={10} onChange={(event)=>{
                intervalRef.current=parseInt(event.target.value);
            }}/>

        </div>


        {start && (

        <div>
            <div>
                <p>Head Position:{headPosition}</p>
            </div>

            <div>
                <p>Gaze Direction:{gazeDirection}</p>
            </div>

        <video ref={videoRef} autoPlay playsInline muted style={{display:'none'}} />
        <Canvas width={640} height={480} source={video!} bbox={bboxRef!} />
        </div>
        )}

        
        </>
    )
}