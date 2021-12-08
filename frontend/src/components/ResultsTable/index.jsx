import React, { useEffect, useState } from 'react';
import Table from 'react-bootstrap/Table';

export default function ResultsTable(props) {
  const resultsCSVStr = props.resultsCSVStr;
  const [results, setResults] = useState([]);
  const headers = ['Id', 'CDR1_nogaps', 'CDR2_nogaps', 'CDR3_nogaps', 'CDRS_IP', 'CDRS_HP', 'logistic_regression_onehot_CDRS', 'logistic_regression_3mer_CDRS','cnn_20','rnn_20','rnn_20_full','cnn_full_10','logistic_regression_onehot_CDRS_full','logistic_regression_3mer_CDRS_full'];

  useEffect(() => {
    const fetchResultsData = () => {
      if (!resultsCSVStr) {
        return;
      }
      let resultsCSVArr = resultsCSVStr.trim().split('\n');
      let rowIndex = 0;
      const allHeaders = resultsCSVArr[rowIndex].split(',');
      rowIndex++;

      var resultsArr = [];
      for (rowIndex; rowIndex < resultsCSVArr.length; rowIndex++) {
        let row = resultsCSVArr[rowIndex].split(',');
        var rowDict = {};
        for (let colIndex = 0; colIndex < allHeaders.length; colIndex++) {
          if (headers.indexOf(allHeaders[colIndex]) >= 0) {
            rowDict[allHeaders[colIndex]] = row[colIndex];
          }
        }
        resultsArr.push(rowDict);
      };
      setResults(resultsArr);
    };
    fetchResultsData();
    // eslint-disable-next-line
  }, [resultsCSVStr]);

  return (
    <>
      <Table striped bordered>
        <tbody>
          <tr>
          {
            headers.map( (header) =>
              <td
                key={`header_${header}`}
              >
                {header}
              </td>
            )
          }
          </tr>
          {
            results &&
            results.map( (row, rowIndex) =>
              <tr
                key={`row_${rowIndex}`}
              >
              {
                headers.map( (header) =>
                  <td
                    key={`row_${rowIndex}_col_${headers.indexOf(header)}`}
                  >
                    {row[header]}
                  </td>
                )
              }
              </tr>
            )
          }
        </tbody>
      </Table>
    </>
  )
}
