def parse_java(code):
    import javalang
    from javalang.ast import Node

    def get_token(node):
        token = ''
        if isinstance(node, str):
            token = node.replace('\"|\'', '')
        elif isinstance(node, set):
            token = 'Modifier'  # node.pop()
        elif isinstance(node, Node):
            token = node.__class__.__name__

        return token

    def get_children(root):
        if isinstance(root, Node):
            children = root.children
        elif isinstance(root, set):
            children = list(root)
        else:
            children = []

        def expand(nested_list):
            for item in nested_list:
                if isinstance(item, list):
                    for sub_item in expand(item):
                        yield sub_item
                elif item:
                    yield item

        return list(expand(children))

    def get_sequence(node, sequence):
        token, children = get_token(node), get_children(node)
        sequence.append(token)

        for child in children:
            get_sequence(child, sequence)

    tokens = javalang.tokenizer.tokenize(code)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    seq = []
    get_sequence(tree, seq)
    return ' '.join(seq)


# f = """
# 	public String getServiceStatus(UID arg0) throws RemoteException {
#
#
#
#
# 		if (getService().isActive()){}
# 			// NON EPO CODE : 04/04/2005, Return service OK, line 2423
# 		else {}
# 			// NON EPO CODE : 04/04/2005, Return service KO, line 2425
# 	}
#
# """
# print(parse_java(f))
