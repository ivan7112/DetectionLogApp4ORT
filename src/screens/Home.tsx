
import { useNavigate } from "react-router"
export default function Home(){
    const navigate=useNavigate()

    return (
    <>
    <div>
        <h1>Sample App for Detection</h1>
        <h2>Camera is needed in order to work</h2>
    </div>   
    
    <div>
        <button onClick={()=>{
            navigate("/detect");
            
        }}>Start Now!</button>
    </div>   
    
    </>)
}