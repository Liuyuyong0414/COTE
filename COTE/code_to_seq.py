from edit import DiffTokenizer, empty_token_filter, construct_diff_sequence_with_con


def operate_seq(code1, code2):
    diff_tokenizer = DiffTokenizer(token_filter=empty_token_filter)
    a_tokens, b_tokens = diff_tokenizer(code1, code2)
    change_seqs = construct_diff_sequence_with_con(a_tokens, b_tokens)
    return change_seqs

    # first_non_equal_index = None
    # last_non_equal_index = None

    # for i, (a_tokens, b_tokens, op) in enumerate(change_seqs):
    #     if op != 'equal':
    #         if first_non_equal_index is None:
    #             first_non_equal_index = i
    #         last_non_equal_index = i
    # if first_non_equal_index is None:
    #     return []
    # else:
    #     return change_seqs[first_non_equal_index:last_non_equal_index + 1]

# Test

# a = "public static RowFactory createDefault() {\n    return withRowTypeFactory(new DefaultRowTypeFactory());\n  }"
# b = "public static RowFactory createDefault() {\n    return withSchemaFactory(new DefaultSchemaFactory());\n  }"
# change_seqs = operate_seq(a, b)
# for change_seq in change_seqs:
#     print(change_seq)
