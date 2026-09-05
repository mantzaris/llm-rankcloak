#!/usr/bin/env ruby
# Applies the limited readability repairs needed after latexdiff:
# (1) display pre-document abstract replacements as marked body text,
# (2) separate the old and new titles so their markup does not overlap, and
# (3) enable flexible line breaking in this author-review artifact.
kind, old_path, new_path, diff_path = ARGV
abort "usage: repair_diff.rb main|supp old.tex new.tex diff.tex" unless diff_path && %w[main supp].include?(kind)

extract_abstract = lambda do |path|
  source = File.binread(path)
  match = source.match(/\\begin\{abstract\}\s*(.*?)\s*\\end\{abstract\}/m)
  raise "abstract not found in #{path}" unless match
  match[1].gsub(/\s+/, " ").strip
end

old_abstract = extract_abstract.call(old_path)
new_abstract = extract_abstract.call(new_path)
diff = File.binread(diff_path)

old_title, new_title =
  if kind == "main"
    [
      "RankCloak: Hiding Synthetic Cryptographic Artifacts in Plain English with Language Models",
      "RankCloak Conceals the Surface Form of Synthetic Cryptographic Artifacts in Language Model Generated Text"
    ]
  else
    [
      "Supplementary Information for RankCloak: Hiding Synthetic Cryptographic Artifacts in Plain English with Language Models",
      "RankCloak Conceals the Surface Form of Synthetic Cryptographic Artifacts in Language Model Generated Text"
    ]
  end

title_pattern =
  if kind == "main"
    /^\\title\{.*\}\n/
  else
    /^\\title\{.*?\n\n(?=\\author)/m
  end
title_blocks = diff.scan(title_pattern).length
raise "expected one title block, found #{title_blocks}" unless title_blocks == 1
title_markup = "\\title{\\DIFdel{#{old_title}}\\\\[0.6em]\\DIFadd{#{new_title}}}"
title_markup += kind == "main" ? "\n" : "\n\n"
diff.sub!(title_pattern) { title_markup }

abstract_blocks = diff.scan(/\\begin\{abstract\}.*?\\end\{abstract\}/m).length
raise "expected one abstract block, found #{abstract_blocks}" unless abstract_blocks == 1
marked_abstract = <<~TEX.chomp
  \\begin{abstract}
  \\textbf{Deleted V2 abstract}\\par
  \\DIFdelbegin\\DIFdel{#{old_abstract}}\\DIFdelend
  \\par\\medskip
  \\textbf{Added final V3 abstract}\\par
  \\DIFaddbegin\\DIFadd{#{new_abstract}}\\DIFaddend
  \\end{abstract}
TEX
diff.sub!(/\\begin\{abstract\}.*?\\end\{abstract\}/m) { marked_abstract }

marker = "\\begin{document}\n"
raise "document marker missing or nonunique" unless diff.scan(marker).length == 1
diff.sub!(marker) { "#{marker}\\setlength{\\emergencystretch}{3em}\n\\sloppy\n" }

figure_width = "\\includegraphics[width=\\textwidth]"
diff.gsub!(figure_width, "\\includegraphics[width=0.97\\textwidth]")
diff.gsub!(/[ \t]+$/, "")

File.binwrite(diff_path, diff)
