

import { useEffect, useRef, type CanvasHTMLAttributes } from "react"



type CanvasProps = {
  width: number;
  height: number;
  source: HTMLVideoElement;
};


export default function Canvas( {width, height, source}:CanvasProps){
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
            }
            raf = requestAnimationFrame(loop); //Vuelve a ejecutar loop en el borrado de pantalla
        };
        
        raf = requestAnimationFrame(loop);

        return () => cancelAnimationFrame(raf); //Función de limpieza, detiene el bucle (Al cambiar source)
  }, [source]);
    
    return (
        <>
        <canvas width={width} height={height} ref={ref}></canvas>
        </>
    )
}