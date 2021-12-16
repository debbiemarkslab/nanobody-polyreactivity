export function faqPageBlurb() {
  return (
    <>
      This webserver tool aligns nanobodies according to IMGT numbering and supplies users with biochemical properties of nanobodies including isoelectric point, hydrophobicity, CDR sequences (IMGT definition), CDR lengths, potential liabilities including the existence of glycosylation sites, and computational predictions of polyreactivity scores by one-hot and 3-mer logistic regression models. For the program to function correctly, nanobody sequences must be entered in fasta format.
      <br/><br/><br/>
      <u><b>Frequently Asked Questions</b></u>
      <br/><br/>
      Q: How do I properly enter a nanobody sequence into the webserver?
      <br/>
      A: Please use fasta formatting for sequence entry. For example:
      <br/><br/>
      >nanobody A02’
      <br/>
      <p style={{'wordBreak': 'break-all' }}>
      QVQLVESGGGLVQAGGSLRLSCAASGIIFYVYAMGWYRQAPGKERELVASISTGGSTNYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAADAGVYVISYLVDYWGQGTQVTVSS
      </p>
      Note that the nanobody sequence must be entered on a new line underneath the nanobody name.
      <br/>
      <br/>
      <hr/>
      <br/>
      Q: What are some typical values of algorithm predictions for low, medium, and high polyreactive nanobodies?
      <br/>
      A: Higher values from the algorithm correlate with lower predicted nanobody polyreactivity. Predictions for nanobody A02’ (low polyreactivity), D06 (medium polyreactivity), and E10’ (high polyreactivity) are:
      <br/>
      		&emsp;Nanobody A02’: 2.0779 (one-hot), 2.6798 (3-mer)
          <br/>
      		&emsp;Nanobody D06: -1.067 (one-hot), 0.1891 (3-mer)
          <br/>
          &emsp;Nanobody E10’: -2.7289 (one-hot), -1.8218 (3-mer)
      <br/>
      <br/>
      <hr/>
      <br/>
      Q: Can I enter multiple nanobody sequences at once into the webserver?
      <br/>
      A: Yes, simply enter each nanobody sequence you want to evaluate using fasta formatting in a list. For example:
      <br/><br/>
      >nanobody A02’
      <br/>
      <p style={{'wordBreak': 'break-all' }}>
      QVQLVESGGGLVQAGGSLRLSCAASGIIFYVYAMGWYRQAPGKERELVASISTGGSTNYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAADAGVYVISYLVDYWGQGTQVTVSS
      <br/>
      >nanobody D06
      <br/>
      QVQLVESGGGLVQAGGSLRLSCAASGRIFGYYAMGWYRQAPGKERELVAVIRGGVSTNYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCNARRYWAFNAYSKYDYWGQGTQVTVSS
      <br/>
      >nanobody E10’
      <br/>
      QVQLVESGGGLVQAGGSLRLSCAASGRIFSHYAMGWYRQAPGKEREFVAAISADGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAARKYYRTNGYWGQGTQVTVSS
      </p>
      <br/>
      <hr/>
      <br/>
      Q: What data are the models used in this webserver trained on?
      <br/>
      A: The models are trained on the expanded deep sequencing data, which comprises 1,221,800 unique non-polyreactive clones and 1,058,842 unique polyreactive clones.
      <br/>
      <br/>
      <hr/>
      <br/>
      Q: Can this webserver evaluate types of antibodies other than nanobodies?
      <br/>
      A: No, this webserver is only intended to estimate the biochemical properties and polyreactivity of nanobodies.
      <br/>
      <br/>
      <hr/>
      <br/>
      Q: How can I predict mutations to make in my nanobody sequence to lower its estimated polyreactivity?
      <br/>
      A: Refer to Figure 3c in the manuscript for position-dependent contributions of amino acids to polyreactivity, as determined by the one-hot logistic regression algorithm. Amino acids colored in blue are correlated with lower polyreactivity, while amino acids colored in red contribute to higher polyreactivity.
      <br/>
      <br/>
      <hr/>
      <br/>
      Please cite the following pre-print when publishing results acquired using this webserver tool:
      <br/><br/>
      Harvey, E.P.*; Shin, J.*; Skiba, M.A.*; Nemeth, G.R.; Hurley, J.D.; Wellner; A.; Miranda, V.G.; Min, J.K.; Liu, C.C.; Marks, D.S. †; Kruse, A.C.† An in silico method to assess antibody-fragment polyreactivity. BioRxiv 2021.
      <br/><br/>
      *Equal contributors listed in alphabetical order
      <br/><br/>
      †Correspondence and requests for materials should be sent to Debora S. Marks (<a href='mailto:debora_marks@hms.harvard.edu'>debora_marks@hms.harvard.edu</a>) or Andrew C. Kruse (<a href='mailto:andrew_kruse@hms.harvard.edu'>andrew_kruse@hms.harvard.edu</a>)
    </>
  )
}
