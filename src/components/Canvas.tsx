

import { useEffect, useRef } from "react"
import type { BBox } from "../utils/processOutput";



type CanvasProps = {
  width: number;
  height: number;
  source: HTMLVideoElement;
  bboxHead: React.RefObject<BBox | null>;
  bboxHandGestures:React.RefObject<BBox | null>;
};


export default function Canvas( {width, height, source, bboxHead,bboxHandGestures}:CanvasProps){
    const ref = useRef<HTMLCanvasElement>(null);

    

    

  
     useEffect(() => {
       
        const canvas = ref.current;
        if (!canvas || !source) return;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        let raf = 0; //Id petición animación

        const loop = () => {
            if (source.readyState >= 2) { //Cuando haya un fotograma cargado
                ctx.drawImage(source, 0, 0, canvas.width, canvas.height); //Fotograma actual

                const headBox=bboxHead.current;
                const handGestureBox=bboxHandGestures.current;
                ctx.strokeStyle="blue";
                

                if (headBox){
                    ctx.strokeRect(headBox.x,headBox.y,headBox.w,headBox.h);
                    ctx.strokeText(headBox.classification + `: ${headBox.score.toFixed(2)}`,headBox.x,headBox.y-5);
                }

                if(handGestureBox){
                    ctx.strokeRect(handGestureBox.x,handGestureBox.y,handGestureBox.w,handGestureBox.h);
                    ctx.strokeText(handGestureBox.classification + `: ${handGestureBox.score.toFixed(2) }` ,handGestureBox.x,handGestureBox.y-5);
                }
                
            }
            raf = requestAnimationFrame(loop); //Vuelve a ejecutar loop en el borrado de pantalla
        };
        
        raf = requestAnimationFrame(loop);

        return () => cancelAnimationFrame(raf); //Función de limpieza, detiene el bucle 
  }, [source]);
    
    return (
        <>
        <canvas width={width} height={height} ref={ref}></canvas>
        </>
    )
}