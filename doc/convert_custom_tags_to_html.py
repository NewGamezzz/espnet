import glob
import os  # noqa
import re

import configargparse

ALL_HTML_TAGS = [
    "a",
    "abbr",
    "acronym",
    "address",
    "applet",
    "area",
    "article",
    "aside",
    "audio",
    "b",
    "base",
    "basefont",
    "bdi",
    "bdo",
    "big",
    "blockquote",
    "body",
    "br",
    "button",
    "canvas",
    "caption",
    "center",
    "cite",
    "code",
    "col",
    "colgroup",
    "data",
    "datalist",
    "dd",
    "del",
    "details",
    "dfn",
    "dialog",
    "dir",
    "div",
    "dl",
    "dt",
    "em",
    "embed",
    "fieldset",
    "figcaption",
    "figure",
    "font",
    "footer",
    "form",
    "frame",
    "frameset",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "head",
    "header",
    "hgroup",
    "hr",
    "html",
    "i",
    "iframe",
    "img",
    "input",
    "ins",
    "kbd",
    "label",
    "legend",
    "li",
    "link",
    "main",
    "map",
    "mark",
    "menu",
    "meta",
    "meter",
    "nav",
    "noframes",
    "noscript",
    "object",
    "ol",
    "optgroup",
    "option",
    "output",
    "p",
    "param",
    "picture",
    "pre",
    "progress",
    "q",
    "rp",
    "rt",
    "ruby",
    "s",
    "samp",
    "script",
    "search",
    "section",
    "select",
    "small",
    "source",
    "span",
    "strike",
    "strong",
    "style",
    "sub",
    "summary",
    "sup",
    "svg",
    "table",
    "tbody",
    "td",
    "template",
    "textarea",
    "tfoot",
    "th",
    "thead",
    "time",
    "title",
    "tr",
    "track",
    "tt",
    "u",
    "ul",
    "var",
    "video",
    "wbr",
]

LANGUAGE_TAG_SET = [
    ("default", "text"),
    ("pycon", "python"),
    ("cd", "text"),
]


def get_parser():
    parser = configargparse.ArgumentParser(
        description="Convert custom tags to markdown",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root",
        type=str,
        help="source python files that contain get_parser() func",
    )
    return parser


def replace_custom_tags(content):
    # Regex to find tags and their content
    tag_pattern = re.compile(r"<(?!!--)([^>]+)>")

    def replace_tag(match):
        tag_name = match.group(1)
        if len(tag_name) > 50:
            # heuristics to ignore tags with too long names
            # This might occur with image tags, since they have image data
            # in base64 format.
            return match.group(0)

        if tag_name.split()[0] not in ALL_HTML_TAGS or (
            len(tag_name.split()) > 1 and "=" not in tag_name
        ):
            return f"&lt;{tag_name}&gt;"

        end_tag_pattern = re.compile(f"</{tag_name.split()[0]}>")
        end_tag_match = end_tag_pattern.search(content, match.end())
        if not end_tag_match:
            return f"&lt;{tag_name}&gt;"
        return match.group(0)

    return tag_pattern.sub(replace_tag, content)


def replace_string_tags(content):
    # Regex to find tags and their content
    tag_pattern = re.compile(r"['|\"]<(?!\/)(.+?)(?!\/)>['|\"]")

    def replace_tag(match):
        tag_name = match.group(1)
        if len(tag_name) > 50:
            # heuristics to ignore tags with too long names
            # This might occur with image tags, since they have image data
            # in base64 format.
            return match.group(0)
        if tag_name.split()[0] not in ALL_HTML_TAGS or (
            len(tag_name.split()) > 1 and "=" not in tag_name
        ):
            return f"'&lt;{tag_name}&gt;'"

        end_tag_pattern = re.compile(f"</{tag_name.split()[0]}>")
        end_tag_match = end_tag_pattern.search(content, match.end())
        if not end_tag_match:
            return f"'&lt;{tag_name}&gt;'"
        return match.group(0)

    return tag_pattern.sub(replace_tag, content)


def replace_language_tags(content):
    for label, lang in LANGUAGE_TAG_SET:
        content = content.replace(f"```{label}", f"```{lang}")

    return content


def _split_fenced_code_blocks(text):
    segments = []
    fence_re = re.compile(r"^[ \t]*([`~]{3,}).*$")
    in_fence = False
    fence_marker = None
    buffer = []

    for line in text.splitlines(keepends=True):
        if not in_fence:
            match = fence_re.match(line)
            if match:
                if buffer:
                    segments.append((False, "".join(buffer)))
                    buffer = []
                in_fence = True
                fence_marker = match.group(1)[0]
                buffer.append(line)
            else:
                buffer.append(line)
        else:
            buffer.append(line)
            match = fence_re.match(line)
            if match and match.group(1)[0] == fence_marker:
                segments.append((True, "".join(buffer)))
                buffer = []
                in_fence = False
                fence_marker = None

    if buffer:
        segments.append((in_fence, "".join(buffer)))

    return segments


def _split_inline_code(text):
    segments = []
    pattern = re.compile(r"(`+)([^`\n]*?)\1")
    last = 0
    for match in pattern.finditer(text):
        if match.start() > last:
            segments.append((False, text[last : match.start()]))
        segments.append((True, match.group(0)))
        last = match.end()
    if last < len(text):
        segments.append((False, text[last:]))
    return segments


def replace_tags_skip_code(content):
    content = replace_language_tags(content)
    fenced_segments = _split_fenced_code_blocks(content)
    output = []
    for is_code_block, block in fenced_segments:
        if is_code_block:
            output.append(block)
            continue
        inline_segments = _split_inline_code(block)
        for is_inline_code, seg in inline_segments:
            if is_inline_code:
                output.append(seg)
            else:
                seg = replace_string_tags(seg)
                seg = replace_custom_tags(seg)
                output.append(seg)
    return "".join(output)


if __name__ == "__main__":
    # parser
    args = get_parser().parse_args()

    for md in glob.glob(f"{args.root}/*.md", recursive=True):
        with open(md, "r") as f:
            content = f.read()

        # Replace the "" and "" with "&lt;" and "&gt;", respectively
        # if the tag is not in ALL_HTML_TAGS and does not have its end tag
        # we need to apply this two functions because
        # there are custom tags like: "<custom-tag a='<type>' b='<value>' />"
        content = replace_tags_skip_code(content)

        with open(md, "w") as f:
            f.write(content)

    for md in glob.glob(f"{args.root}/**/*.md", recursive=True):
        with open(md, "r") as f:
            content = f.read()

        # Replace the "" and "" with "&lt;" and "&gt;", respectively
        # if the tag is not in ALL_HTML_TAGS
        content = replace_tags_skip_code(content)

        with open(md, "w") as f:
            f.write(content)
