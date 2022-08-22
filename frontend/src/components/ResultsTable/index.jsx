import React, { useEffect, useState } from 'react';
import Table from 'react-bootstrap/Table';

export default function ResultsTable(props) {
  const resultsCSVStr = props.resultsCSVStr;
  const [results, setResults] = useState([]);
  // const headers = ['Id', 'CDR1_nogaps', 'CDR2_nogaps', 'CDR3_nogaps', 'CDRS_IP', 'CDRS_HP', 'origFACS lr onehot','origFACS lr 3mers','origFACS cnn onehot','origFACS rnn onehot','deepFACS lr onehot','deepFACS lr 3mer','deepFACS cnn onehot','deepFACS rnn onehot'];
  const headers = ['Id', 'CDR1_nogaps', 'CDR2_nogaps', 'CDR3_nogaps', 'isoelectric point', 'hydrophobicity', 'origFACS lr onehot','deepFACS lr onehot'];

  useEffect(() => {
    const fetchResultsData = () => {
      if (!resultsCSVStr) {
        return;
      }
      let resultsCSVArr = resultsCSVStr.trim().split('\n');
      let rowIndex = 0;
      const allHeaders = resultsCSVArr[rowIndex].split(',');
      rowIndex++;

      var rowsToDisplay = 31;
      if (resultsCSVArr.length < rowsToDisplay) {
        rowsToDisplay = resultsCSVArr.length;
      }

      var resultsArr = [];
      for (rowIndex; rowIndex < rowsToDisplay; rowIndex++) {
        let row = resultsCSVArr[rowIndex].split(',');
        var rowDict = {};
        for (let colIndex = 0; colIndex < allHeaders.length; colIndex++) {
          if (headers.indexOf(allHeaders[colIndex]) >= 0) {

            // Round to 2 decimal places
            var valueToDisplay = row[colIndex];
            if (!isNaN(parseFloat(row[colIndex]))) {
              valueToDisplay = parseFloat(row[colIndex]);
              valueToDisplay = +valueToDisplay.toFixed(2);
            }
            rowDict[allHeaders[colIndex]] = valueToDisplay;

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
