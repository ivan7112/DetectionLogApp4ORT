
import * as ort from "onnxruntime-web";
import Canvas from "../components/Canvas";
import { useEffect, useState, useRef } from "react";
import { videoToTensor, videoToHeadPoseTensor} from "../utils/videototensor";
import { classifyGazeDirection, classifyHeadPosition, type Classification } from "../utils/classifications";
import { eyeCropToTensor } from "../utils/cropEye";
import createCsv from "../utils/createCsv";
import { processHead } from "../utils/processOutput";
import { processHandGesture } from "../utils/processOutput";



export default function Detect(){

    interface CsvRow{
        HH:string,
        MM:string,
        SS:string,
        head_position:string,
        gaze_direction:string,
        hand_gesture:string
    }

    const videoRef = useRef<HTMLVideoElement>(null);
    const [video, setVideo] = useState<HTMLVideoElement | null>(null);

    const [start, setStart]=useState(false);
    const [headPosition,setHeadPosition]=useState("");
    const [gazeDirection, setGazeDirection]=useState("");
    const [handGesture,setHandGesture]=useState("");

    const startRef=useRef(start);
    const intervalRef=useRef(10);
    const frameRef=useRef(0);
    const busyRef=useRef(false);
    const sessionRef = useRef<ort.InferenceSession | null>(null);
    const sessionHeadPositionRef=useRef<ort.InferenceSession | null>(null);
    const sessionGazeDirectionRef=useRef<ort.InferenceSession | null>(null);
    const sessionHandGestureRef=useRef<ort.InferenceSession | null>(null);

    const bboxHeadref = useRef<ReturnType<typeof processHead>>(null);
    const bboxHandGesturesRef=useRef<ReturnType<typeof processHandGesture>>(null);

    const handGestureRef=useRef<string>("");


    //CSV Rows. The format will be HH,MM,SS,head_position,gaze_direction
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

     const getSessionHandGesture=async  ()=>{
        if (!sessionHandGestureRef.current) sessionHandGestureRef.current=await ort.
            InferenceSession.create('/hand_gestures.onnx',{executionProviders: ["webgpu", "wasm"]});
        
        return sessionHandGestureRef.current;
    }
    

    useEffect(() => {
        let stream: MediaStream | null = null;
        let cancelled = false;
        

        (async () => { //Open camera
            if(!start) return;
                const s = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                if (cancelled) {
                    s.getTracks().forEach((t) => t.stop());
                    return;
                }
                stream = s;
                const v = videoRef.current!;
                v.srcObject = s;
                await v.play();
                setVideo(v); // Now canvas starts drawing

            
        })();

        return () => {
        cancelled = true;
        stream?.getTracks().forEach((t) => t.stop());
        };
  }, [start]);

    useEffect(() => { startRef.current = start; }, [start]);
    

    const headDetectionInference=async  (video:HTMLVideoElement) =>{
        //Stores detections in csvRows until the camera is closed

        let rowCsv:CsvRow={
            HH:"",
            MM:"",
            SS:"",
            head_position:"",
            gaze_direction:"",
            hand_gesture:""

        };

        const {tensor, letterbox}=videoToTensor(video);
        const session=await getSession();
        const handSession=await getSessionHandGesture();

        //Detectors are called and their results are processed to bboxes
        const result=await session.run({[session.inputNames[0]]:tensor});
        const hand_gesture_result=await handSession.run({[session.inputNames[0]]:tensor});

        const output=result[session.outputNames[0]];
        const hand_gesture_outpout=hand_gesture_result[session.outputNames[0]];




        bboxHeadref.current=processHead(output, letterbox); //tensor to bbox 
        bboxHandGesturesRef.current=processHandGesture(hand_gesture_outpout,letterbox);

        if (bboxHandGesturesRef.current){
            setHandGesture(bboxHandGesturesRef.current.classification);
            handGestureRef.current=bboxHandGesturesRef.current.classification;
        }
        //Classify if we detect a head
        if (bboxHeadref.current){


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
                gaze_direction:"",
                hand_gesture:handGestureRef.current
            }

            if (headPosition==="center"){
                const tensor=eyeCropToTensor(video, bboxHeadref.current);

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

    
    useEffect(()=>{ //Main loop
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
    return ()=>cancelAnimationFrame(rafid); //Without this loop would continue even upon dismount
    },[]);


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
                csvRows.current=[]; //Reset CSV rows
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

            <div>
                <p>Hand Gesture:{handGestureRef.current}</p>
            </div>

        <video ref={videoRef} autoPlay playsInline muted style={{display:'none'}} />
        <Canvas width={640} height={480} source={video!} bboxHead={bboxHeadref!} bboxHandGestures={bboxHandGesturesRef!} />
        </div>
        )}

        
        </>
    )
}