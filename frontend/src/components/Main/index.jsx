import { Route, Routes } from 'react-router-dom';

import HomePage from '../../pages/HomePage';
import NotFoundPage from '../../pages/NotFoundPage';
import ResultsPage from '../../pages/ResultsPage';

export default function Main() {
  return (
    <Routes>
      <Route exact path='/' element={<HomePage/>}></Route>
      <Route exact path='/results' element={<ResultsPage/>}></Route>
      <Route exact path='/results/:resultsId' element={<ResultsPage/>}></Route>
      <Route element={<NotFoundPage/>} />
    </Routes>
  )
}
