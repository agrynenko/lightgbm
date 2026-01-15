from pathlib import Path

def convert(input_filename, out_data_filename, out_query_filename, out_query_filename2):
	input = open(input_filename,"r")
	output_feature = open(out_data_filename,"w")
	output_query = open(out_query_filename,"w")
	output_query2 = open(out_query_filename2,"w")
	cur_cnt = 0
	cur_doc_cnt = 0
	last_qid = -1
	while True:
		line = input.readline()
		if not line:
			break
		tokens = line.split(' ')
		tokens[-1] = tokens[-1].strip()
		label = tokens[0]
		qid = int(tokens[1].split(':')[1])
		if qid != last_qid:
			if cur_doc_cnt > 0:
				output_query.write(str(cur_doc_cnt) + '\n')
				output_query2.write(str(cur_doc_cnt) + '\n')
				cur_cnt += 1
			cur_doc_cnt = 0
			last_qid = qid
		cur_doc_cnt += 1
		output_feature.write(label+' ')
		output_feature.write(' '.join(tokens[2:]) + '\n')
	output_query.write(str(cur_doc_cnt) + '\n')
	output_query2.write(str(cur_doc_cnt) + '\n')
	
	input.close()
	output_query.close()
	output_feature.close()
	output_query2.close()

in_folder = 'data/Fold1'
out_folder = 'data'

Path(out_folder).mkdir(parents=True, exist_ok=True)

convert(
    f"${in_folder}/train.txt",
    f"${out_folder}/mslr.train",
    f"${out_folder}/mslr.train.query",
    f"${out_folder}/mslr.train.group"
)

convert(
    f"${in_folder}/val.txt",
    f"${out_folder}/mslr.val",
    f"${out_folder}/mslr.val.query",
    f"${out_folder}/mslr.val.group"
)

convert(
    f"${in_folder}/test.txt",
    f"${out_folder}/mslr.test",
    f"${out_folder}/mslr.test.query",
    f"${out_folder}/mslr.test.group"
)
