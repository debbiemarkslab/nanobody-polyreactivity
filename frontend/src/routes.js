
const baseUrl = (process.env.REACT_APP_BACKEND_BASE_URL ? process.env.REACT_APP_BACKEND_BASE_URL : '')
export const scatterHistUrl = (identifier) => {
  return `${baseUrl}/plots/${identifier}/`
};

export const scoreSequencesUrl = `${baseUrl}/score_sequences/`;
export const scoreSequencesFileUrl = `${baseUrl}/score_sequences_file/`;

export const scoresUrl = (identifier) => {
  return `${baseUrl}/scores/${identifier}/`
};
