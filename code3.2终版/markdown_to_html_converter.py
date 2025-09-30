#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown到HTML转换器
支持数学公式、代码高亮、图片、表格等特殊元素的完整转换
"""

import markdown
import os
import re
from pathlib import Path

class MarkdownToHTMLConverter:
    """Markdown到HTML转换器类"""
    
    def __init__(self):
        """初始化转换器"""
        self.setup_markdown_extensions()
        
    def setup_markdown_extensions(self):
        """设置Markdown扩展"""
        self.extensions = [
            'markdown.extensions.tables',      # 表格支持
            'markdown.extensions.fenced_code', # 代码块支持
            'markdown.extensions.codehilite',  # 代码高亮
            'markdown.extensions.toc',         # 目录支持
            'markdown.extensions.attr_list',   # 属性列表
            'markdown.extensions.def_list',    # 定义列表
            'markdown.extensions.footnotes',   # 脚注支持
            'markdown.extensions.md_in_html',  # HTML中的Markdown
        ]
        
        self.extension_configs = {
            'markdown.extensions.codehilite': {
                'css_class': 'highlight',
                'use_pygments': True,
                'noclasses': False,
            },
            'markdown.extensions.toc': {
                'permalink': True,
                'toc_depth': 6,
            }
        }
    
    def create_html_template(self, title="复赛大题三完整解决方案", content=""):
        """创建HTML模板"""
        html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    
    <!-- MathJax for mathematical formulas -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true,
                processEnvironments: true
            }},
            options: {{
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
            }}
        }};
    </script>
    
    <!-- Prism.js for code highlighting -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" rel="stylesheet" />
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet" />
    
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fafafa;
        }}
        
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: #2c3e50;
            margin-top: 2em;
            margin-bottom: 1em;
            font-weight: 600;
        }}
        
        h1 {{
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            font-size: 2.5em;
        }}
        
        h2 {{
            border-bottom: 2px solid #e74c3c;
            padding-bottom: 8px;
            font-size: 2em;
        }}
        
        h3 {{
            border-left: 4px solid #f39c12;
            padding-left: 15px;
            font-size: 1.5em;
        }}
        
        h4 {{
            color: #8e44ad;
            font-size: 1.3em;
        }}
        
        /* 表格样式 */
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }}
        
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        
        tr:hover {{
            background-color: #e8f4f8;
        }}
        
        /* 代码样式 */
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
        }}
        
        pre {{
            background-color: #2d3748;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        pre code {{
            background-color: transparent;
            padding: 0;
            color: inherit;
        }}
        
        /* 列表样式 */
        ul, ol {{
            margin: 15px 0;
            padding-left: 30px;
        }}
        
        li {{
            margin: 8px 0;
        }}
        
        /* 引用样式 */
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 20px 0;
            padding: 15px 20px;
            background-color: #f8f9fa;
            font-style: italic;
        }}
        
        /* 图片样式 */
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 20px 0;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }}
        
        /* 强调文本 */
        strong {{
            color: #e74c3c;
            font-weight: 600;
        }}
        
        em {{
            color: #8e44ad;
        }}
        
        /* 分隔线 */
        hr {{
            border: none;
            height: 2px;
            background: linear-gradient(to right, #3498db, #e74c3c);
            margin: 40px 0;
        }}
        
        /* 目录样式 */
        .toc {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        
        .toc ul {{
            list-style-type: none;
            padding-left: 20px;
        }}
        
        .toc a {{
            text-decoration: none;
            color: #3498db;
        }}
        
        .toc a:hover {{
            text-decoration: underline;
        }}
        
        /* 数学公式样式 */
        .MathJax {{
            font-size: 1.1em !important;
        }}
        
        /* 响应式设计 */
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
            
            .container {{
                padding: 20px;
            }}
            
            h1 {{
                font-size: 2em;
            }}
            
            h2 {{
                font-size: 1.7em;
            }}
            
            table {{
                font-size: 0.9em;
            }}
            
            th, td {{
                padding: 8px;
            }}
        }}
        
        /* 打印样式 */
        @media print {{
            body {{
                background-color: white;
                color: black;
            }}
            
            .container {{
                box-shadow: none;
                padding: 0;
            }}
            
            h1, h2 {{
                page-break-after: avoid;
            }}
            
            table {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {content}
    </div>
    
    <!-- Prism.js for code highlighting -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
</body>
</html>"""
        return html_template
    
    def preprocess_markdown(self, content):
        """预处理Markdown内容"""
        # 处理数学公式 - 确保LaTeX公式被正确识别
        # 将行内数学公式标记转换为MathJax格式
        content = re.sub(r'(?<!\\)\$([^$]+)\$', r'\\($1\\)', content)
        
        # 将块级数学公式标记转换为MathJax格式
        content = re.sub(r'(?<!\\)\$\$([^$]+)\$\$', r'\\[$1\\]', content)
        
        # 处理图片路径 - 确保相对路径正确
        content = re.sub(r'!\[([^\]]*)\]\(figures/([^)]+)\)', r'![\\1](figures/\\2)', content)
        
        # 处理表格中的粗体文本
        content = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\\1</strong>', content)
        
        return content
    
    def postprocess_html(self, html_content):
        """后处理HTML内容"""
        # 为代码块添加语言类
        html_content = re.sub(
            r'<pre><code class="language-python">',
            r'<pre><code class="language-python">',
            html_content
        )
        
        # 确保表格有正确的类名
        html_content = re.sub(r'<table>', r'<table class="table">', html_content)
        
        # 为图片添加alt属性和标题
        html_content = re.sub(
            r'<img src="([^"]+)" alt="([^"]*)"',
            r'<img src="\\1" alt="\\2" title="\\2"',
            html_content
        )
        
        return html_content
    
    def convert_file(self, input_file, output_file=None):
        """转换单个Markdown文件为HTML"""
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        
        # 确定输出文件路径
        if output_file is None:
            output_file = input_path.with_suffix('.html')
        
        output_path = Path(output_file)
        
        # 读取Markdown内容
        with open(input_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # 预处理Markdown内容
        processed_content = self.preprocess_markdown(markdown_content)
        
        # 创建Markdown解析器
        md = markdown.Markdown(
            extensions=self.extensions,
            extension_configs=self.extension_configs
        )
        
        # 转换为HTML
        html_body = md.convert(processed_content)
        
        # 后处理HTML内容
        html_body = self.postprocess_html(html_body)
        
        # 获取文档标题
        title_match = re.search(r'^#\s+(.+)$', markdown_content, re.MULTILINE)
        title = title_match.group(1) if title_match else "文档"
        
        # 创建完整的HTML文档
        full_html = self.create_html_template(title=title, content=html_body)
        
        # 写入HTML文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"✅ 转换完成: {input_file} -> {output_file}")
        return output_path

def main():
    """主函数"""
    # 设置输入和输出文件路径
    input_file = r"f:\25DIA\code3.2\复赛大题三完整解决方案.md"
    output_file = r"f:\25DIA\code3.2\复赛大题三完整解决方案.html"
    
    try:
        # 创建转换器实例
        converter = MarkdownToHTMLConverter()
        
        # 执行转换
        result_path = converter.convert_file(input_file, output_file)
        
        print(f"\n🎉 HTML文件已成功生成: {result_path}")
        print(f"📁 文件大小: {result_path.stat().st_size / 1024:.1f} KB")
        
        # 检查生成的HTML文件
        with open(result_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        print(f"📄 HTML内容长度: {len(html_content)} 字符")
        print(f"🔗 包含的图片数量: {html_content.count('<img')}")
        print(f"📊 包含的表格数量: {html_content.count('<table')}")
        print(f"💻 包含的代码块数量: {html_content.count('<pre><code')}")
        
        print("\n✨ 转换特性:")
        print("  • 支持MathJax数学公式渲染")
        print("  • 支持Prism.js代码语法高亮")
        print("  • 响应式设计，适配移动设备")
        print("  • 美观的表格和图片样式")
        print("  • 完整的CSS样式和交互效果")
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()