
import * as ort from "onnxruntime-web";

export default function Detect(){

    const headDetectionInference=async  (tensor:ort.Tensor) =>{

        try{

        const session=await ort.InferenceSession.create('../yolo-models/detectors/head_detector.onnx');
        
        const feeds={image:tensor};

        const result=await session.run(feeds);


        }catch(e){
            console.log("Error during inference: ",e)
        }

    }



    return (
        <>
        <p>WIP as of now</p>
        </>
    )
}