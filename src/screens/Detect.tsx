
import * as ort from "onnxruntime-web";
import Canvas from "../components/Canvas";
import { useEffect, useState, useRef } from "react";
import { videoToTensor, videoToHeadPoseTensor, videoToEmotionTensor} from "../utils/videototensor";
import { classifyEmotion, classifyGazeDirection, classifyHeadPosition, EMOTION_NAMES, topNclasses, type Classification } from "../utils/classifications";
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
        hand_gesture:string,
        emotions:string
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
    const sessionHandGestureRef=useRef<ort.InferenceSession | null>(null);
    const sessionEmotionRef=useRef<ort.InferenceSession | null>(null);

    const bboxHeadref = useRef<ReturnType<typeof processHead>>(null);
    const bboxHandGesturesRef=useRef<ReturnType<typeof processHandGesture>>(null);

    const handGestureRef=useRef<string>("");
    const emotionsRef=useRef<Array<string>>([]);



    //CSV Rows. The format will be HH,MM,SS,head_position,gaze_direction
    const csvRows = useRef<Array<CsvRow>>([]);


    //Send data to Backend
    const BACKEND_URL="ws://localhost:8000/uploadlog"
    const wsref=useRef<WebSocket>(null)
   


    const sendDatatoBack= async ()=>{
        console.log("something");
       
    }



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

    const getSessionEmotion= async ()=>{
        if (!sessionEmotionRef.current) sessionEmotionRef.current=await ort.InferenceSession
        .create('/emotion_classifier.onnx',{executionProviders:["webgpu","wasm"]});

        return sessionEmotionRef.current;

    }

    useEffect(()=>{
        const ws=new WebSocket(BACKEND_URL);
        wsref.current=ws

        ws.onopen= ()=>{
            console.log("Conection opened");
            sendDatatoBack()
        }
        ws.onclose= ()=> console.log("Conection closed");
        ws.onerror= (err)=> console.log("An error has ocurred: ", err)
        ws.onmessage= (msg) =>console.log(msg)

        return ()=>ws.close()

    },[])
    

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
    

    const DetectionInference=async  (video:HTMLVideoElement) =>{
        //Stores detections in csvRows until the camera is closed

        let rowCsv:CsvRow={
            HH:"",
            MM:"",
            SS:"",
            head_position:"",
            gaze_direction:"",
            hand_gesture:"",
            emotions:"[]"

        };

        const {tensor, letterbox}=videoToTensor(video);
        const headsession=await getSession();
        const handSession=await getSessionHandGesture();

        //Detectors are called and their results are processed to bboxes
        const head_result=await headsession.run({[headsession.inputNames[0]]:tensor});
        const hand_gesture_result=await handSession.run({[headsession.inputNames[0]]:tensor});

        const output=head_result[headsession.outputNames[0]];
        const hand_gesture_outpout=hand_gesture_result[headsession.outputNames[0]];




        bboxHeadref.current=processHead(output, letterbox); //tensor to bbox 
        bboxHandGesturesRef.current=processHandGesture(hand_gesture_outpout,letterbox);

        if (bboxHandGesturesRef.current){
            
            handGestureRef.current=bboxHandGesturesRef.current.classification;
        }
        //Classify if we detect a head
        if (bboxHeadref.current){

            //Head pose Classification
            const headPoseTensor=videoToHeadPoseTensor(video).tensor;
            const sessionHeadPose=await getSessionHeadPosition();
            const resultHeadPosition=await sessionHeadPose.run({[sessionHeadPose.inputNames[0]]:headPoseTensor})
            const output2=resultHeadPosition[sessionHeadPose.outputNames[0]];

            const headPoseClass:Classification=classifyHeadPosition(output2);

            setHeadPosition(headPoseClass.label);

            //Emotion Classification
            const emotionTensor=videoToEmotionTensor(video).tensor;
            const sessionEmotion=await getSessionEmotion();
            const resultEmotion= await sessionEmotion.run({[sessionEmotion.inputNames[0]]:emotionTensor});
            const outputEmotion=resultEmotion[sessionEmotion.outputNames[0]];

            const emotionClass:Classification=classifyEmotion(outputEmotion);
            
            emotionsRef.current=topNclasses(1,emotionClass.probs,EMOTION_NAMES);
            

          



            //Csv row creation
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
                hand_gesture:handGestureRef.current,
                emotions:"[" + emotionsRef.current.join(',') + "]"
            }

            if (headPosition==="center"){
                const eyetensor=eyeCropToTensor(video, bboxHeadref.current);

                const sessionGazeDirection=await  getSessiongazeDirection();
                const resultGazeDirection= await sessionGazeDirection.run({[ sessionGazeDirection.inputNames[0]]:eyetensor});
                const output3= resultGazeDirection[sessionGazeDirection.outputNames[0]];

                const gazeDirectionClass=classifyGazeDirection(output3);
                setGazeDirection(gazeDirectionClass.label);

                rowCsv={...rowCsv, gaze_direction:gazeDirectionClass.label};
            }

            csvRows.current.push(rowCsv);

            
        }

        const logBackend={log:Object.values(rowCsv).join(',')}
        const auxcanvas=document.createElement("canvas")
        const ctxaux=auxcanvas.getContext("2d")
        const videoaux=videoRef.current
        
        if (ctxaux && videoaux && bboxHeadref.current){
            auxcanvas.width = videoaux.videoWidth;
            auxcanvas.height = videoaux.videoHeight;
            ctxaux.drawImage(videoaux,0,0,videoaux.videoWidth,videoaux.videoHeight);
            auxcanvas.toBlob((blob)=>{
                if (!blob) return;
                wsref.current?.send(JSON.stringify(logBackend));
                wsref.current?.send(blob);

            },"image/jpg",1.0);
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

            await DetectionInference(v!);
            
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

            {emotionsRef.current && emotionsRef.current.length && (
                <>
                <div>
                    <p>Posible emotions: {emotionsRef.current.join(',')}</p>
                </div>
                        
                </>
            )

            }

        <video ref={videoRef} autoPlay playsInline muted style={{display:'none'}} />
        <Canvas width={640} height={480} source={video!} bboxHead={bboxHeadref!} bboxHandGestures={bboxHandGesturesRef!} />
        </div>
        )}

        
        </>
    )
}