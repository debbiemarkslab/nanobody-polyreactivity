from app.common.score_new_sequences import score_sequences

# filename = "/home/ec2-user/nanobody-polyreactivity/backend/inputs/0eeba69e-5212-42b0-9d4d-a921b0a0617d.fa"
# identifier = "0eeba69e-5212-42b0-9d4d-a921b0a0617d"
filename = "/home/ec2-user/nanobody-polyreactivity/backend/inputs/fb5875ca-b5a7-4f70-a81b-6aad6f477079.fa"
identifier = "fb5875ca-b5a7-4f70-a81b-6aad6f477079.fa"
score_sequences(filename,identifier)
print('finished!')