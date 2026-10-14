import { Route, Routes } from 'react-router'
import './App.css'
import Home from './screens/Home'
import Detect from './screens/Detect'

function App() {

  return (
    <>
    <Routes>

      <Route path='/' element={<Home/>}/>
      <Route path='/detect' element={<Detect/>}/>
    </Routes>
      
    </>
  )
}

export default App
